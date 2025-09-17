export default function LoadingDots() {
  return (
    <div className="flex gap-1 pl-2 text-blue-600">
      <span className="animate-bounce">●</span>
      <span className="animate-bounce delay-150">●</span>
      <span className="animate-bounce delay-300">●</span>
    </div>
  );
}
