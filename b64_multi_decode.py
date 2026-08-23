import sys, base64
text = sys.stdin.read().strip()
try:
  text = base64.b64decode(text).decode()
  text = base64.b64decode(text).decode()
except:
  pass
print(text, end='')