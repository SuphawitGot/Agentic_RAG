"""Watch the agent work, step by step:

    uv run python -m ragsale.agent "Which camera does Demo Line A use?"
"""
import argparse
import json

from .graph import RECURSION_LIMIT, build_graph, make_model, start


def _short(value, limit=160):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + '...'


def describe(node, update):
    """One readable line or more for each node's update."""
    if update.get('result') and node == 'agent':
        return ['agent: out of budget, stopping']
    lines = []
    for message in update.get('messages', []):
        if node == 'agent':
            for call in message.tool_calls:
                lines.append(f"agent -> {call['name']}({_short(call['args'])})")
            if not message.tool_calls:
                lines.append(f'agent said (no tool): {_short(message.content)}')
        elif node == 'tools':
            content = json.loads(message.content)
            if content['ok']:
                found = sum(len(p['excerpts']) for p in content['projects'] if isinstance(p['excerpts'], list))
                lines.append(f"tools: {message.name} found {found} excerpts, {content['searches_left']} searches left")
            else:
                lines.append(f"tools: {message.name} refused: {content['error']}")
        else:
            lines.append(f'{node}: {_short(message.content)}')
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('question')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--project', action='append', default=[], dest='projects',
                        help='Limit the search to this project ID (repeatable).')
    args = parser.parse_args()

    state, early = start(args.question, args.top_k, args.projects)
    result = early
    if state is not None:
        # stream_mode="updates" yields {node name: what that node returned} after every step.
        for step in build_graph(make_model()).stream(state, {'recursion_limit': RECURSION_LIMIT},
                                                     stream_mode='updates'):
            for node, update in step.items():
                for line in describe(node, update or {}):
                    print(line)
                if update and update.get('result'):
                    result = update['result']

    print('\n' + result['answer'])
    for source in result.get('sources', []):
        meta = source['metadata']
        print(f"[{source['citation']}] {meta.get('project_name')} - {meta.get('filename')}, page {meta.get('page')}")
    for option in result.get('project_options', []):
        print(f"- {option['project_name']} ({option['project_id']})")


if __name__ == '__main__':
    main()
