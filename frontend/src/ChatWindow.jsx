import React, { useEffect, useRef } from "react";

export default function ChatWindow({ messages }) {
  const ref = useRef();
  useEffect(() => {
    ref.current.scrollTop = ref.current.scrollHeight;
  }, [messages]);

  return (
    <div id="chat-window" className="chat-window" ref={ref}>
      {messages.map((m, i) => (
        <div key={i} className={`message ${m.who} ${m.error ? "error" : ""}`}>
          {m.text}
        </div>
      ))}
    </div>
  );
}
