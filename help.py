from pyautd3.link.ethercrab import Status

# Statusの中身をすべて表示する (これが一番わかりやすい)
print("--- Statusのメンバー一覧 ---")
for s in Status:
    print(s)
print("----------------------------")

# help()機能を使う (詳しい説明（docstring）があれば表示される)
help(Status)