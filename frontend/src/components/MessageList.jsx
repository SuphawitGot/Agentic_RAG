import { useLayoutEffect, useRef, useState } from 'react';
import SourceList from './SourceList';

export default function MessageList({ messages = [], searching = false, pendingId, onChooseProjects }) {
  const viewport = useRef(null);
  const followLatest = useRef(true);
  const [showLatest, setShowLatest] = useState(false);

  function jumpToLatest() {
    const node = viewport.current;
    node.scrollTop = node.scrollHeight;
    followLatest.current = true;
    setShowLatest(false);
  }

  useLayoutEffect(() => {
    // Follow new replies only when the reader is already near the bottom.
    if (followLatest.current || messages.at(-1)?.role === 'user') jumpToLatest();
  }, [messages, searching]);

  function onScroll() {
    const node = viewport.current;
    const nearBottom = node.scrollHeight - node.scrollTop - node.clientHeight < 80;
    followLatest.current = nearBottom;
    setShowLatest(!nearBottom);
  }

  return <div className="conversation-wrap">
    <section ref={viewport} className="conversation" aria-label="Conversation" role="log"
      aria-live="polite" tabIndex={0} onScroll={onScroll}>
      {messages.length === 0 ? <div className="empty">
        <span className="chat-mark" aria-hidden="true">R</span>
        <h2>What would you like to know?</h2>
        <p>Ask about your documents. Answers include sources you can check.</p>
      </div> : messages.map(message => <article key={message.id} className={`chat-message ${message.role}`}>
        <span className="message-author">{message.role === 'user' ? 'You' : 'RagSale · Qwen'}</span>
        <div className="message-bubble">
          <p className="message">{message.content}</p>
          {message.projectOptions?.length > 0 && <div className="project-choices" aria-label="Choose a project">
            {message.projectOptions.map(project => <button key={project.project_id}
              disabled={searching || pendingId !== message.id}
              onClick={() => onChooseProjects([project])}>
              <strong>{project.project_name}</strong>
              <span>{project.documents.map(d => d.filename).join(', ')}</span>
              {message.projectOptions.filter(p => p.project_name === project.project_name).length > 1 &&
                <span>Upload {project.project_id.slice(-8)}</span>}
            </button>)}
            {message.projectOptions.length > 1 && message.projectOptions.length <= 12 &&
              <button disabled={searching || pendingId !== message.id}
                onClick={() => onChooseProjects(message.projectOptions)}>Compare these projects</button>}
          </div>}
          <SourceList sources={message.sources} />
        </div>
      </article>)}
      {searching && <article className="chat-message assistant">
        <span className="message-author">RagSale · Qwen</span>
        <div className="message-bubble pending">Searching documents and preparing your answer…</div>
      </article>}
    </section>
    {showLatest && <button className="latest-button" onClick={jumpToLatest}>↓ Latest messages</button>}
  </div>;
}
