import React from "react";

export default function Sources({ sources = [] }) {
  if (!sources.length)
    return <p style={{ color: "#9aa7b2", margin: 8 }}>No sources found.</p>;

  return (
    <>
      {sources.map((s, i) => (
        <div key={i} className="source-card">
          <div className="source-title">
            {s.rank}. {s.case_name} ({s.year})
          </div>
          <div className="source-meta">
            {s.court || ""}
            {s.citation ? " • " + s.citation : ""}
          </div>
          <div className="source-meta">{(s.area_of_law || []).join(", ")}</div>
          <div style={{ marginTop: 6 }}>{s.summary || ""}</div>
        </div>
      ))}
      <div className="footer-links">
        Tip: visit <a href="http://127.0.0.1:8000/docs" target="_blank">API Docs</a>
      </div>
    </>
  );
}
