interface SemanticFlowProps {
  agent:  string;
  action: string;
  target: string;
}



export default function SemanticFlow({ agent, action, target }: SemanticFlowProps) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '0',
        flexWrap: 'wrap',
        rowGap: '16px',
      }}
    >
      {/* Agent node */}
      <Node label={agent} type="agent" />

      {/* Arrow */}
      <Arrow />

      {/* Action node */}
      <ActionNode label={action} />

      {/* Arrow */}
      <Arrow />

      {/* Target node */}
      <Node label={target} type="target" />
    </div>
  );
}

function Node({ label, type }: { label: string; type: 'agent' | 'target' }) {
  const isAgent = type === 'agent';
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '6px',
      }}
    >
      <div
        style={{
          background: isAgent ? '#000' : 'rgba(0,0,0,0.07)',
          color: isAgent ? '#fff' : '#000',
          border: isAgent ? '2px solid #000' : '2px solid rgba(0,0,0,0.15)',
          borderRadius: '12px',
          padding: '10px 22px',
          fontFamily: 'var(--font-serif)',
          fontSize: '1.05rem',
          fontWeight: 400,
          letterSpacing: '-0.01em',
          whiteSpace: 'nowrap',
          transition: 'transform 0.2s',
          cursor: 'default',
        }}
        onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.transform = 'translateY(-2px)'; }}
        onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.transform = 'translateY(0)'; }}
      >
        {label}
      </div>
      <span
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '0.62rem',
          letterSpacing: '0.14em',
          color: 'var(--accent)',
          textTransform: 'uppercase',
        }}
      >
        {isAgent ? 'Agent' : 'Target'}
      </span>
    </div>
  );
}

function ActionNode({ label }: { label: string }) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '6px',
      }}
    >
      <div
        style={{
          background: 'rgba(0,0,0,0.04)',
          border: '2px dashed rgba(0,0,0,0.2)',
          borderRadius: '99px',
          padding: '10px 24px',
          fontFamily: 'var(--font-sans)',
          fontSize: '0.88rem',
          fontWeight: 600,
          color: '#000',
          letterSpacing: '0.04em',
          whiteSpace: 'nowrap',
          textTransform: 'uppercase',
        }}
      >
        {label}
      </div>
      <span
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '0.62rem',
          letterSpacing: '0.14em',
          color: 'var(--accent)',
          textTransform: 'uppercase',
        }}
      >
        Action
      </span>
    </div>
  );
}

function Arrow() {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        padding: '0 8px',
        marginBottom: '20px',
        color: 'rgba(111,111,111,0.45)',
        fontSize: '1.4rem',
        lineHeight: 1,
      }}
    >
      →
    </div>
  );
}
