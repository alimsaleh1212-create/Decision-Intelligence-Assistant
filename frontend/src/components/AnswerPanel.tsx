interface Props {
  ragAnswer: string | null;
  nonRagAnswer: string | null;
  isLoading: boolean;
}

function SkeletonLines() {
  return (
    <div className="answer-skeleton">
      <div className="skeleton-line" style={{ width: "92%" }} />
      <div className="skeleton-line" style={{ width: "78%" }} />
      <div className="skeleton-line" style={{ width: "85%" }} />
      <div className="skeleton-line" style={{ width: "60%" }} />
    </div>
  );
}

interface CardProps {
  badge: string;
  badgeClass: string;
  answer: string | null;
  isLoading: boolean;
  delay: number;
}

function AnswerCard({ badge, badgeClass, answer, isLoading, delay }: CardProps) {
  return (
    <div className="answer-card" style={{ animationDelay: `${delay}ms` }}>
      <div className="answer-card-header">
        <span className={`answer-badge ${badgeClass}`}>{badge}</span>
      </div>
      {isLoading && !answer ? (
        <SkeletonLines />
      ) : answer ? (
        <div className="answer-body">{answer}</div>
      ) : (
        <div className="answer-body" style={{ color: "var(--text-muted)", fontStyle: "italic" }}>
          Submit a query to see the answer.
        </div>
      )}
    </div>
  );
}

export function AnswerPanel({ ragAnswer, nonRagAnswer, isLoading }: Props) {
  return (
    <div>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "var(--text-secondary)",
          marginBottom: 12,
        }}
      >
        Generated answers
      </div>
      <div className="answers-grid">
        <AnswerCard
          badge="RAG"
          badgeClass="rag"
          answer={ragAnswer}
          isLoading={isLoading}
          delay={0}
        />
        <AnswerCard
          badge="No Context"
          badgeClass="non-rag"
          answer={nonRagAnswer}
          isLoading={isLoading}
          delay={80}
        />
      </div>
    </div>
  );
}
