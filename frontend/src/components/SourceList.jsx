export default function SourceList({ sources = [] }) {
  if (!sources.length) return null;
  return <details className="source-details"><summary>Sources ({sources.length})</summary>
    <ol className="sources">{sources.map((source, index) => <li key={source.id ?? index}>
      <strong>[{source.citation ?? index + 1}] {source.metadata?.filename ?? source.metadata?.source ?? `Source ${index + 1}`}</strong>
      {source.metadata?.page != null && <span> · Page {source.metadata.page}</span>}
      {source.metadata?.row_index != null && <span> · Dataset row {source.metadata.row_index}</span>}
      <p className="message">{source.text}</p>
      <small>Cosine distance: {source.distance.toFixed(4)} (lower is closer)</small>
    </li>)}</ol>
  </details>;
}
