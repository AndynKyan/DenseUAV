| Tập | Nguồn | Số lớp | Ảnh drone | Ảnh satellite |
|---|---|---|---|---|
| train | lớp lưới (pseudo) | 27 | 324 biến thể + 8 ảnh UAV thật đã gộp | 27 |
| train | UAV thật, lớp riêng | 2 | 2 | 2 |
| train | **tổng** | **29** | **334** | **29** |
| val | lớp lưới (pseudo) | 12 | 144 biến thể + 4 ảnh UAV thật đã gộp | 12 |
| val | UAV thật, lớp riêng | 0 | 0 | 0 |
| val | **tổng** | **12** | **148** | **12** |
| test | UAV thật (query) | 14 | 14 | – |
| bị loại | UAV thật | 20 (buffer 18, nodata 2) | | |

Khoảng cách tới tâm lớp gần nhất (m):

| Tập | Tâm tính | min | trung vị |
|---|---|---|---|
| train | chỉ lớp lưới | 14.00 | 14.00 |
| train | lưới + lớp UAV thật riêng | 14.00 | 14.00 |
| val | chỉ lớp lưới | 14.00 | 14.00 |
| val | lưới + lớp UAV thật riêng | 14.00 | 14.00 |

file ẩn đã xoá: 0
