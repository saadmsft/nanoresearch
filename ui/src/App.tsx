import { ChatPanel } from "@/components/ChatPanel";
import { StatusPanel } from "@/components/StatusPanel";

export default function App() {
  return (
    <div className="grid h-screen grid-cols-[minmax(0,1fr)_340px] gap-0">
      <ChatPanel />
      <StatusPanel />
    </div>
  );
}
