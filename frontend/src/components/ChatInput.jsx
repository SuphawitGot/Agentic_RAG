import { useState } from 'react';

export default function ChatInput({ onSend, disabled = false, busy = disabled }) {
  const [question, setQuestion] = useState('');
  function submit(event) {
    event.preventDefault();
    if (disabled || !question.trim()) return;
    onSend(question.trim());
    setQuestion('');
  }
  return (
    <form className="chat-composer" onSubmit={submit}>
      <label className="sr-only" htmlFor="question">Your question</label>
      <textarea id="question" value={question} onChange={event => setQuestion(event.target.value)}
        placeholder="Ask about your documents…" disabled={disabled} rows={2}
        onKeyDown={event => {
          if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            event.currentTarget.form.requestSubmit();
          }
        }} maxLength={2000} />
      <button disabled={disabled || !question.trim()} type="submit">{busy ? 'Working…' : 'Send ↑'}</button>
    </form>
  );
}
