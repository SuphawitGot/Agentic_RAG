import { useEffect, useState } from 'react';
import { askQuestion } from '../api/chat';
import { getHealth } from '../api/client';
import ThemeToggle from '../components/ThemeToggle';
import FileUpload from '../components/FileUpload';
import ChatInput from '../components/ChatInput';
import MessageList from '../components/MessageList';

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState('');
  const [lastRequest, setLastRequest] = useState(null);
  const [pending, setPending] = useState(null);
  const [status, setStatus] = useState('Checking backend…');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setStatus('Checking backend…');
    const timeout = setTimeout(() => controller.abort(), 5000);
    let active = true;
    getHealth(controller.signal)
      .then(() => { if (active) setStatus('Backend connected'); })
      .catch(() => { if (active) setStatus('Backend unavailable'); })
      .finally(() => clearTimeout(timeout));
    return () => { active = false; clearTimeout(timeout); controller.abort(); };
  }, [attempt]);

  async function sendQuestion(question, scope = {}, displayText = question, retry = false) {
    if (searching) return;
    setSearching(true);
    setError('');
    setLastRequest({ question, scope, displayText });
    if (!retry) setMessages(current => [...current, { id: crypto.randomUUID(), role: 'user', content: displayText }]);
    try {
      const data = await askQuestion(question, 3, scope);
      const id = crypto.randomUUID();
      setPending(data.type === 'clarification' ? { id, question } : null);
      setMessages(current => [...current, {
        id, role: 'assistant',
        content: data.answer,
        sources: data.sources,
        projectOptions: data.type === 'clarification' ? data.project_options : [],
      }]);
    } catch (err) {
      setError(err instanceof TypeError ? 'Cannot reach the backend. Check that FastAPI is running.' : err.message);
    } finally {
      setSearching(false);
    }
  }

  function chooseProjects(projects) {
    if (!pending || searching) return;
    sendQuestion(pending.question, { selected_project_ids: projects.map(p => p.project_id) },
      projects.length > 1 ? `Compare: ${projects.map(p => p.project_name).join(', ')}` : projects[0].project_name);
  }

  return <main>
    <header><div className="header-brand"><p className="eyebrow">DOCUMENT ASSISTANT</p><h1>RagSale</h1><ThemeToggle /></div>
      <div className="header-actions"><div><p role="status">{status}</p><button onClick={() => setAttempt(value => value + 1)}>Check connection</button></div></div>
    </header>
    <FileUpload />
    <div className="panel chat-panel">
      <div className="chat-heading"><h2>Chat with your documents</h2><span>Answers with sources</span></div>
      <MessageList messages={messages} searching={searching} pendingId={pending?.id} onChooseProjects={chooseProjects} />
      <div className="chat-footer">
      <p className="chat-caption">Check the sources to verify an answer.</p>
      {error && <div role="alert"><p>{error}</p><button disabled={searching || !lastRequest}
        onClick={() => sendQuestion(lastRequest.question, lastRequest.scope, lastRequest.displayText, true)}>Retry</button></div>}
      {pending && <p role="status">Choose a project above to continue your original question.{' '}
        <button disabled={searching} onClick={() => { setPending(null); setError(''); }}>Ask a different question</button></p>}
      <ChatInput onSend={question => sendQuestion(question)} disabled={searching || Boolean(pending)} busy={searching} />
      </div>
    </div>
  </main>;
}
