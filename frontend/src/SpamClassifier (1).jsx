import { useState, useEffect, useRef } from "react";

const STYLES = ``;

const SAMPLES = {
  spam: [
    "CONGRATULATIONS! You've been selected as TODAY'S LUCKY WINNER! Claim your $5,000 cash prize NOW! Click here: bit.ly/win-cash. Limited time offer! Act immediately or lose your prize FOREVER! Reply with your bank details to receive payment.",
    "URGENT: Your account will be suspended! Verify your information immediately at secure-login.suspicious-domain.com. Failure to respond within 24 hours will result in permanent account termination. Click now!",
    "Make $10,000/week from home! No experience needed! Our revolutionary system guarantees income! Thousands already earning MASSIVE profits! Limited spots available. Buy now for only $49.99 and start earning TODAY!!!",    
  ],
  legit: [
    "Hi Sarah, just following up on our meeting from Tuesday. I've attached the revised project timeline as discussed. Could you review it and let me know if the deadlines work for your team? Best regards, Michael",
    "Team, the quarterly review is scheduled for Friday at 2pm in Conference Room B. Please bring your Q4 reports and come prepared to discuss the roadmap for next quarter. Lunch will be provided.",
    "Your order #12345 has been shipped and is expected to arrive by Thursday, March 5th. You can track your package using the tracking number: 1Z999AA1012345678. Thank you for shopping with us!",
  ],
};

export default function SpamClassifier() {
  const [email, setEmail] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({ total: 0, spam: 0, legit: 0 });
  const [confAnim, setConfAnim] = useState(0);
  const textRef = useRef();

  useEffect(() => {
    if (result) {
      setConfAnim(0);
      setTimeout(() => setConfAnim(result.confidence), 50);
    }
  }, [result]);

  const analyze = async () => {
    if (!email.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(import.meta.env.VITE_API_URL + "/api/model",{
        method:"POST", 
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({email})
      });

      const data = await response.json();
      const text = data.content.map(i => i.text || "").join("");
      const clean = text.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(clean);

      setResult(parsed);
      const isSpam = parsed.classification === "SPAM";
      setHistory(prev => [{
        text: email.substring(0, 60) + (email.length > 60 ? "..." : ""),
        isSpam,
        confidence: parsed.confidence
      }, ...prev.slice(0, 9)]);
      setStats(prev => ({
        total: prev.total + 1,
        spam: prev.spam + (isSpam ? 1 : 0),
        legit: prev.legit + (!isSpam ? 1 : 0)
      }));
    } catch (err) {
      setError("Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const isSpam = result?.classification === "SPAM";

  return (
    <>
      <style>{STYLES}</style>
      <div className="app">
        <div className="grid-bg" />
        <div className="scanline" />

        <div className="content">
          {/* Header */}
          <div className="header">
            {/* <div className="header-tag">AI-Powered Detection</div>   */}
            <h1>SPAM<br />SHIELD</h1>
            <p className="subtitle">Email threat classification </p>
          </div>

          {/* Stats */}
          <div className="stats-bar">
            <div className="stat">
              <span className="stat-value">{stats.total}</span>
              <span className="stat-label">Analyzed</span>
            </div>
            <div className="stat">
              <span className="stat-value" style={{ color: 'var(--danger)' }}>{stats.spam}</span>
              <span className="stat-label">Spam Caught</span>
            </div>
            <div className="stat">
              <span className="stat-value">{stats.legit}</span>
              <span className="stat-label">Legit Passed</span>
            </div>
          </div>

          {/* Input */}
          <div className="main-card">
            <div className="card-label"></div> {/*// Email Content   */}
            <textarea
              ref={textRef}
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="Paste email content here for analysis..."
            />
            <div className="char-count">{email.length} chars</div>

            <div className="samples">
              <span style={{ fontSize: '10px', color: 'var(--muted)', letterSpacing: '1px', alignSelf: 'center' }}>SAMPLES:</span>
              {SAMPLES.spam.map((s, i) => (
                <button key={i} className="sample-btn spam-sample" onClick={() => setEmail(s)}>
                  Spam #{i + 1}
                </button>
              ))}
              {SAMPLES.legit.map((s, i) => (
                <button key={i} className="sample-btn" onClick={() => setEmail(s)}>
                  Legit #{i + 1}
                </button>
              ))}
            </div>

            <button
              className={`analyze-btn ${loading ? 'loading' : ''}`}
              onClick={analyze}
              disabled={loading || !email.trim()}
            >
              {loading ? (
                <span className="loader">
                  <span /><span /><span />
                  &nbsp;&nbsp;ANALYZING
                </span>
              ) : 'ANALYZE EMAIL'}
            </button>

            {error && <div className="error-box">⚠ {error}</div>}
          </div>

          {/* Result */}
          {result && (
            <div className={`result-card ${isSpam ? 'spam' : 'legit'}`} style={{ marginBottom: 32 }}>
              <div className="result-header">
                <div className="verdict">
                  <div className="verdict-icon">{isSpam ? '⚠' : '✓'}</div>
                  <div className="verdict-text">
                    <div className="verdict-label">Classification</div>
                    <div className="verdict-value">{isSpam ? 'SPAM' : 'LEGITIMATE'}</div>
                  </div>
                </div>
                <div className="confidence-wrap">
                  <div className="confidence-label">Confidence</div>
                  <div className="confidence-bar">
                    <div className="confidence-fill" style={{ width: `${confAnim}%` }} />
                  </div>
                  <div className="confidence-pct">{result.confidence}%</div>
                </div>
              </div>

              <div className="result-body">
                <div className="indicators-title"></div> {/*// Detection Signals */}
                <div className="indicators">
                  {result.indicators?.map((ind, i) => (
                    <div key={i} className="indicator">
                      <div className="indicator-dot" />
                      {ind}
                    </div>
                  ))}
                </div>

                {/* <div className="tags">
                  {result.tags?.map((tag, i) => {
                    const cls = isSpam
                      ? (i === 0 ? 'tag-red' : 'tag-yellow')
                      : 'tag-green';
                    return <span key={i} className={`tag ${cls}`}>{tag}</span>;
                  })}
                </div> */}
              </div>
            </div>
          )}

          {/* History */}
          <div className="history-card">
            <div className="history-header">
              <div className="history-title">Analysis History</div>
              {history.length > 0 && (
                <button className="clear-btn" onClick={() => setHistory([])}>Clear</button>
              )}
            </div>

            {history.length === 0 ? (
              <div className="empty-history">No analyses yet — scan your first email above</div>
            ) : (
              <div className="history-list">
                {history.map((item, i) => (
                  <div key={i} className="history-item">
                    <span className={`history-badge ${item.isSpam ? 'badge-spam' : 'badge-legit'}`}>
                      {item.isSpam ? 'SPAM' : 'LEGIT'}
                    </span>
                    <span className="history-text">{item.text}</span>
                    <span className="history-conf" style={{ color: item.isSpam ? 'var(--danger)' : 'var(--accent)' }}>
                      {item.confidence}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}