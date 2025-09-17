export default function HomePage() {
  return (
    <main className="relative flex min-h-screen flex-col items-center justify-center text-center px-4 sm:px-6 
                     bg-gradient-to-br from-green-50 via-white to-blue-50 overflow-hidden">
      <div className="absolute -top-32 -right-32 w-60 h-60 sm:w-80 sm:h-80 bg-green-300 rounded-full blur-3xl opacity-20 animate-pulse" />
      <div className="absolute -bottom-32 -left-32 w-72 h-72 sm:w-96 sm:h-96 bg-blue-400 rounded-full blur-3xl opacity-20 animate-pulse" />

      <div className="relative z-10 w-full max-w-6xl animate-fadeIn">
        <div className="flex flex-col items-center gap-10 lg:flex-row lg:items-center lg:gap-16">
          <div className="flex-1 text-center lg:text-left">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold bg-gradient-to-r from-green-600 to-blue-700 
                           bg-clip-text text-transparent drop-shadow-sm leading-snug">
              กยศ. Chatbot
            </h1>
            <p className="mt-4 sm:mt-5 text-base sm:text-lg lg:text-xl text-gray-700 leading-relaxed">
              ผู้ช่วยอัจฉริยะสำหรับตอบคำถามเกี่ยวกับกองทุนเงินให้กู้ยืมเพื่อการศึกษา (กยศ.)
              <br className="hidden sm:block" />
              ใช้ง่าย รวดเร็ว ตอบตรงประเด็น
            </p>

            <a
              href="/chat"
              className="mt-6 sm:mt-8 inline-block rounded-full bg-gradient-to-r from-green-600 to-blue-700 
                         px-8 sm:px-12 py-3 sm:py-4 text-base sm:text-lg font-bold text-white shadow-lg 
                         hover:scale-110 hover:shadow-green-400/50 transition-all duration-300"
            >
              เริ่มต้นใช้งาน
            </a>
          </div>

          <div className="flex-1 flex justify-center">
            <img
              src="/home.png"
              alt="กยศ Chatbot Illustration"
              className="w-56 sm:w-72 lg:w-full max-w-md rounded-2xl shadow-xl hover:scale-105 hover:rotate-1 transition-transform duration-500"
            />
          </div>
        </div>

        <div className="mt-12 sm:mt-20 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
          {[
            {
              title: "❓ ตอบคำถาม",
              desc: "ครอบคลุมทุกเรื่องเกี่ยวกับการกู้ยืม กยศ.",
            },
            {
              title: "📄 เอกสารที่ต้องใช้",
              desc: "แนะนำเอกสารที่จำเป็นอย่างครบถ้วน",
            },
            {
              title: "⚡ รวดเร็วทันใจ",
              desc: "ค้นหาข้อมูลได้ในไม่กี่วินาที",
            },
          ].map((f, i) => (
            <div
              key={i}
              className="p-5 sm:p-6 bg-white/80 backdrop-blur-md rounded-2xl shadow-lg 
                         hover:shadow-green-200/70 hover:-translate-y-1 transition-all duration-300"
            >
              <h3 className="text-xl sm:text-2xl font-semibold bg-gradient-to-r from-green-600 to-blue-700 bg-clip-text text-transparent">
                {f.title}
              </h3>
              <p className="mt-2 sm:mt-3 text-gray-600 text-sm sm:text-base leading-relaxed">
                {f.desc}
              </p>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
