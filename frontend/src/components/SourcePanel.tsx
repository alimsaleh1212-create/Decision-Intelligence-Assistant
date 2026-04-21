import type { RetrievedTicket } from "../types";

interface Props {
  tickets: RetrievedTicket[];
}

function scoreClass(score: number): string {
  if (score >= 0.8) return "high";
  if (score >= 0.5) return "medium";
  return "low";
}

export function SourcePanel({ tickets }: Props) {
  if (tickets.length === 0) {
    return (
      <div className="source-section">
        <div className="source-empty">
          <div className="source-empty-icon">◈</div>
          <div className="source-empty-text">No tickets retrieved yet</div>
        </div>
      </div>
    );
  }

  return (
    <div className="source-section">
      <div className="source-section-label">
        Retrieved sources
        <span className="source-count-badge">{tickets.length}</span>
      </div>
      {tickets.map((ticket, i) => (
        <div
          className="ticket-card"
          key={i}
          style={{ animationDelay: `${i * 60}ms` }}
        >
          <div className="ticket-meta">
            <span className="ticket-idx">#{i + 1}</span>
            <span className={`ticket-score ${scoreClass(ticket.score)}`}>
              {(ticket.score * 100).toFixed(1)}% match
            </span>
          </div>
          <div className="ticket-text">{ticket.text}</div>
        </div>
      ))}
    </div>
  );
}
