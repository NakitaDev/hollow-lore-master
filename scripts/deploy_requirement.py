# scripts/deploy.py
import subprocess

# 1. export requirements ไม่มี version
result = subprocess.run(
    ["uv", "export", "--no-hashes", "--no-emit-project", "--format", "requirements-txt"],
    capture_output=True,
    text=True
)

# 2. ลบ version และ gradio ออก
lines = result.stdout.splitlines()
cleaned = []
for line in lines:
    if line.startswith("#") or "gradio" in line or not line.strip():
        continue
    package = line.split("==")[0].strip()
    if package:
        cleaned.append(package)

with open("requirements.txt", "w") as f:
    f.write("\n".join(cleaned))

print("✅ requirements.txt พร้อมแล้ว")

# 3. git push
subprocess.run(["git", "add", "."])
subprocess.run(["git", "commit", "-m", "deploy update"])
subprocess.run(["git", "push", "space", "main"])

print("🚀 Deploy เสร็จแล้ว!")