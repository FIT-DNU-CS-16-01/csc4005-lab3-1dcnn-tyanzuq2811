# GitHub Classroom Guide

## Hướng dẫn nộp bài qua GitHub Classroom

### 1. Accept Assignment
- Bấm link Assignment trên Notion/GitHub Classroom.
- Clone repo về máy.

### 2. Làm bài
- Chạy pipeline theo hướng dẫn trong `README.md`.
- Kết quả lưu trong thư mục `outputs/`.

### 3. Push kết quả
```bash
git add .
git commit -m "Lab 3: MFCC + 1D-CNN baseline"
git push origin main
```

### 4. Kiểm tra CI
- Vào tab Actions trên GitHub để xem kết quả CI.
- CI sẽ kiểm tra cấu trúc repo và chạy smoke test.

### 5. Lưu ý
- **Không push** file `.wav`, `.zip`, thư mục `data/`, `.cache/` lên GitHub.
- Kiểm tra `.gitignore` trước khi push.
