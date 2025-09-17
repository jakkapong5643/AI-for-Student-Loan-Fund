"use client";

import { useState } from "react";
import ChatBubble from "../../components/ChatBubble";
import ChatInput from "../../components/ChatInput";
import LoadingDots from "../../components/LoadingDots";
import { askApi } from "../../lib/api";

export default function ChatPage() {
  const [messages, setMessages] = useState<{ role: string; text: string }[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async (q: string) => {
    if (!q.trim()) return;
    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setLoading(true);
    try {
      const res = await askApi(q);
      setMessages((prev) => [...prev, { role: "assistant", text: res.answer }]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "API Error" },
      ]);
    }
    setLoading(false);
  };

  const clearChat = () => setMessages([]);

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-green-50 via-white to-blue-50">
      <header className="w-full bg-gradient-to-r from-green-600 to-blue-700 text-white shadow-md">
        <div className="max-w-5xl mx-auto flex justify-between items-center px-6 py-4">
          <h1 className="font-bold text-xl drop-shadow">กยศ. Chatbot</h1>
          <button
            onClick={clearChat}
            className="text-sm px-4 py-2 rounded-full bg-white/20 hover:bg-white/30 
                       border border-white/40 shadow-sm transition"
          >
            Clear Chat
          </button>
        </div>
      </header>

      <main className="flex-1 w-full flex justify-center items-center px-4 py-8">
        <div className="max-w-5xl w-full h-[72vh] flex flex-col 
                        bg-white/70 backdrop-blur-md border border-gray-200 
                        rounded-2xl shadow-lg overflow-hidden">
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.map((m, i) => (
              <ChatBubble key={i} role={m.role} text={m.text} />
            ))}
            {loading && <LoadingDots />}
          </div>

          <div className="border-t bg-white/60 backdrop-blur-sm p-4">
            <ChatInput onSend={handleSend} disabled={loading} />
          </div>
        </div>
      </main>

    </div>
  );
}
