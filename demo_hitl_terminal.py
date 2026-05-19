"""
Mô phỏng HITL trên terminal: chạy graph tới interrupt, hỏi approve/reject, resume bằng Command.
Chạy: python demo_hitl_terminal.py
"""
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

load_dotenv()

from graph_builder import get_app_graph  # noqa: E402


def main() -> None:
    graph = get_app_graph()
    tid = str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}
    q = input("Nhập mã CP / câu hỏi: ").strip() or "Tóm tắt ngành ngân hàng Việt Nam"
    print("\n--- stream(start) ---")
    for chunk in graph.stream({"messages": [HumanMessage(content=q)]}, config):
        print(chunk)
    snap = graph.get_state(config)
    print("\ninterrupts:", snap.interrupts)
    if not snap.interrupts:
        print("Không có interrupt (kiểm tra graph / API keys).")
        return
    choice = input("\nDuyệt trước Critic? [y] approve / [n] reject: ").strip().lower()
    if choice == "n":
        fb = input("Feedback: ").strip()
        cmd = Command(resume={"action": "reject", "feedback": fb})
    else:
        cmd = Command(resume={"action": "approve"})
    print("\n--- stream(resume) ---")
    for chunk in graph.stream(cmd, config):
        print(chunk)
    snap2 = graph.get_state(config)
    print("\nKết thúc. next =", snap2.next, "status =", snap2.values.get("status"))


if __name__ == "__main__":
    main()
