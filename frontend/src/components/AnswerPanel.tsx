interface Props {
  ragAnswer: string | null;
  nonRagAnswer: string | null;
  isLoading: boolean;
}

function SkeletonLines() {
  return (
    <div className="answer-skeleton">
      <div className="skeleton-line" style={{ width: "90%" }} />
      <div className="skeleton-line" style={{ width: "76%" }} />
      <div className="skeleton-line" style={{ width: "84%" }} />
      <div className="skeleton-line" style={{ width: "58%" }} />
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
        <div
          className="answer-body"
          style={{ color: "var(--text-muted)", fontStyle: "italic", fontWeight: 300 }}
        >
          Submit a query to see the answer.
        </div>
      )}
    </div>
  );
}

export function AnswerPanel({ ragAnswer, nonRagAnswer, isLoading }: Props) {
  return (
    <div>
      <div className="section-eyebrow">Generated answers</div>
      <div className="answers-grid">
        <AnswerCard
          badge="RAG · Context-augmented"
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
