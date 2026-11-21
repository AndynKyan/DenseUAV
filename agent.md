# Quy tắc làm việc với Claude trong dự án DenseUAV

Dự án này có 2 phiên Claude chạy song song. Mỗi phiên chỉ có một vai trò.
Đầu mỗi phiên, người dùng sẽ khai báo vai trò, ví dụ: "Vai trò: TERMINAL".
Nếu người dùng chưa khai báo, hãy coi đó là phiên EXTENSION (chỉ đọc) và hỏi lại trước khi sửa bất cứ thứ gì.

## Vai trò EXTENSION (phiên trong VS Code): hỏi và đọc

Được làm:
- Đọc, tìm kiếm và giải thích code, log, kết quả huấn luyện.
- Chạy các lệnh chỉ đọc như `ls`, `cat`, `grep`, `git status`, `git diff`, `git log`.
- Đề xuất cách sửa. Hãy viết đề xuất thành một đoạn prompt để người dùng dán sang phiên TERMINAL, nói rõ file, dòng và nội dung cần sửa.

Không được làm:
- Tạo, sửa hoặc xoá bất kỳ file nào trong dự án.
- Chạy lệnh làm thay đổi trạng thái: `git commit`, `git checkout`, `pip install`, `rm`, `mv`, chạy huấn luyện, chạy script ghi file, v.v.

Nếu người dùng yêu cầu sửa trong phiên này, hãy nhắc họ chuyển sang TERMINAL và đưa sẵn prompt cho họ.

## Vai trò TERMINAL (phiên `claude` trong terminal): sửa code và chạy lệnh

Được làm:
- Sửa và tạo file, chạy huấn luyện, chạy test, cài thư viện, dùng git.

Cần làm:
- Trước khi sửa một file, đọc lại phiên bản hiện tại của nó, vì người dùng có thể đã tự sửa file đó.
- Sau khi sửa, tóm tắt ngắn: đã sửa file nào, dòng nào. Như vậy phiên EXTENSION có thể đọc lại và kiểm tra.
- Chỉ commit khi người dùng yêu cầu.

## Quy tắc chung cho cả hai phiên

- Không để hai phiên cùng sửa một file.
- Trả lời bằng tiếng Việt.
- Để tiết kiệm hạn mức, khi chuyển sang việc mới, nhắc người dùng gõ `/clear`.
