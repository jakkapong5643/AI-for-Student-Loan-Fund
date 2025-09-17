export default function AboutPage() {
  return (
    <main className="relative flex min-h-screen items-center justify-center bg-gradient-to-br from-green-50 via-white to-blue-50 px-6 py-12 overflow-hidden">
      <div className="absolute -top-40 -right-40 w-80 h-80 bg-green-300 rounded-full blur-3xl opacity-20 animate-pulse" />
      <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-blue-400 rounded-full blur-3xl opacity-20 animate-pulse" />

      <div className="relative z-10 max-w-3xl w-full bg-white/80 backdrop-blur-md rounded-2xl shadow-lg p-8 animate-fadeIn">
        <h1 className="text-4xl font-extrabold bg-gradient-to-r from-green-600 to-blue-700 
                       bg-clip-text text-transparent drop-shadow-sm mb-6">
          เกี่ยวกับ Chatbot
        </h1>

        <p className="text-lg text-gray-700 leading-relaxed mb-6">
          แชทบอทนี้ถูกพัฒนาขึ้นเพื่อช่วยตอบคำถามที่เกี่ยวข้องกับ{" "}
          <span className="font-semibold">กองทุนเงินให้กู้ยืมเพื่อการศึกษา (กยศ.)</span>{" "}
          โดยใช้เทคโนโลยี <span className="italic">FastAPI + FAISS + bge-m3 + Ollama Large language model (llama3.2-typhoon2-1b)</span> 
          เพื่อให้การค้นหาและตอบคำถามมีความรวดเร็วและใกล้เคียงกับข้อมูลจริงมากที่สุด
        </p>

        <h2 className="text-2xl font-bold text-green-700 mt-6 mb-3">วิธีใช้งาน</h2>
        <ul className="list-disc list-inside space-y-2 text-gray-700">
          <li>พิมพ์คำถามในกล่องข้อความ</li>
          <li>กด <span className="font-semibold">Enter</span> หรือปุ่ม <span className="font-semibold">ส่ง</span></li>
          <li>บอทจะแสดงคำตอบพร้อมเหตุผลและแหล่งข้อมูล</li>
        </ul>

        <h2 className="text-2xl font-bold text-blue-700 mt-8 mb-3">ข้อจำกัด</h2>
        <div className="space-y-2 text-gray-700">
          <p>คำตอบของบอทอาจไม่ถูกต้อง 100%</p>
          <p>ผู้ใช้ควรตรวจสอบข้อมูลจากแหล่งทางการของ กยศ. เพิ่มเติม</p>
        </div>
      </div>
    </main>
  );
}
