# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 11/06/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.18/2.11.0+cpu/8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468; `class_name`="cab"; `rank`=1 ; `score`= 0.510915; `taxonomy_name`="ImageNet-1K"
(Top-5: cab 0.511, minibus 0.164, police_van 0.086, recreational_vehicle 0.054, streetcar 0.048.)
2. Nói về cả bức ảnh. Record chỉ có một lớp xếp hạng cho toàn ảnh.
3. Đi kèm taxonomy ImageNet-1K của checkpoint `yolo11n-cls.pt`, không phải model tự nghĩ ra. Model chỉ chọn trong danh sách có sẵn. Vì vậy cảnh đường phố bị gán vào các lớp xe (cab, minibus…), vì ImageNet không có lớp "cảnh giao thông".
4. Cùng một số ID có thể là tên khác ở taxonomy khác. Tên dễ bị đổi cách viết hoặc ngôn ngữ, còn ID giúp máy đối chiếu chính xác. `taxonomy_name` cho biết ID và tên thuộc danh sách nào.
5. Phải quy định rõ tiêu chí chọn một lớp: ví dụ chọn chủ thể chiếm diện tích lớn nhất hoặc nổi bật nhất, thứ tự ưu tiên giữa các lớp.
6. `score` chỉ là độ tin cậy của model để xếp hạng/lọc, không chứng minh prediction đúng. Ở `traffic`, top-1 chỉ có score 0.51 và là suy đoán trong một taxonomy không phù hợp với cảnh. Ground truth phải do người gán theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person"; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92,  348.92] ; `bbox_width`=113.58; `bbox_height`=279.68
2. Giải thích box bằng lời thường: tọa độ tính theo pixel, gốc (0, 0) ở góc trên-trái ảnh, x tăng sang phải, y tăng xuống dưới. Góc trên-trái của hộp là (385.33, 69.24), góc dưới-phải là (498.92, 348.92), rộng khoảng 113.58 pixel và cao khoảng 279.68 pixel (ảnh `kitchen` kích thước 640 × 427). Trên ảnh, đó là hình chữ nhật cao và hẹp bao quanh người đứng quay lưng.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). ngưỡng 0.20 có **17** prediction; ngưỡng 0.60 có **6** prediction (person ×2, bowl ×2, oven ×2). Ở ngưỡng mặc định 0.35 có 11 prediction.
4. Từ 0.35 xuống 0.20 có thêm 6 box score thấp (spoon ×3, potted plant, dining table, bottle). Reviewer phải kiểm tra từng box thêm: có thật là vật đó không, đúng lớp không, hộp có ôm đúng không, rồi loại box sai. Khối lượng review tăng và dễ lẫn box đúng với box sai. Ngược lại, nâng lên 0.60 thì mất 5 prediction có thật trong ảnh (3 bowl, 2 cup) so với 0.35. Vì vậy ngưỡng chỉ dùng để lọc prediction, không phải quy tắc bỏ vật khi tạo ground truth: annotator vẫn phải gán nhãn cho vật rõ ràng dù model cho score thấp.
5. Mỗi vật thể thuộc taxonomy được vẽ một hộp chữ nhật ôm sát phần nhìn thấy của nó, không chừa viền thừa và không cắt vào vật. Hai vật cùng lớp thì mỗi vật một hộp riêng. Không chỉnh hộp hoặc gán nhãn để khớp với prediction của model.
6. `person` score 0.610931, `bbox_xyxy=[0.12, 263.18, 61.55, 310.78]`, chỉ lộ một phần cánh tay ở mép trái ảnh. Viết thành **quy tắc trong guideline**: vật bị khung ảnh cắt mà vẫn nhận ra lớp thì vẽ hộp ôm phần nhìn thấy, được phép sát mép ảnh. Vì tình huống này lặp lại ở nhiều ảnh nên cần quy tắc chung để các annotator vẽ giống nhau.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-001" ; `class_name`="person"; `score`= 0.899318; số điểm polygon=`polygon_point_count`=348; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. Box chỉ là hình chữ nhật (4 số) nên chứa cả nền xung quanh. Polygon là chuỗi điểm `[x, y]` bám theo biên thật của vật, cho biết hình dạng. Ở `kitchen-001`, box là `[385.45, 66.44, 498.02, 348.58]`, còn polygon dùng 348 điểm bám theo đầu, vai, tay và chân của người đàn ông.
3. Để phân biệt từng vật riêng lẻ trong output của lab, kể cả hai vật cùng lớp (ví dụ `kitchen-002`, `kitchen-003`, `kitchen-006` đều là `bowl` nhưng mỗi cái có ID và polygon riêng).  Nó không phải class ID vì không cho biết vật thuộc lớp nào: lớp nằm ở `class_id` và `class_name`, còn `kitchen-001` chỉ là số thứ tự của một vật trong ảnh. Nó không phải tracking ID vì không theo dõi cùng một vật qua nhiều khung hình hay nhiều ảnh: `kitchen-001` chỉ có nghĩa trong ảnh `kitchen` của lần xuất output này, ảnh khác sẽ đánh số lại.
4. Đặt các điểm polygon dọc mép ngoài phần nhìn thấy của vật, đủ dày để bám góc và đường cong, không lấn sang vật khác, không để lọt nền vào trong. Phần bị vật khác che thì không đưa vào mask của vật bị che.
5. `kitchen-007` (`oven`, score 0.489667) có mask bị thủng và trải xuống vùng sau mặt bàn, biên giữa lò và bàn không rõ. Chọn **guideline**, vì tình huống vật chạm nhau lặp lại ở nhiều ảnh và có thể viết thành quy tắc chung: mỗi vật một polygon bám phần nhìn thấy, mask không chồng nhau và không thủng bên trong vật.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh |  `traffic`: top-1 `cab` 0.51, top-2 `minibus` 0.16; ảnh có cả bus và car nhưng chỉ chọn được một lớp  | Chọn một lớp theo tiêu chí trong guideline; không rõ thì escalation | Lớp có đúng guideline và nhất quán giữa các ảnh không |
| Phát hiện vật thể | Một hộp `[x_min, y_min, x_max, y_max]` + lớp cho mỗi vật | `kitchen`: `person 0.61` bị khung ảnh cắt (`x_min=0.12`); chảo, nồi không có lớp trong COCO-80 | Vẽ hộp ôm sát phần nhìn thấy; không bỏ vật vì score thấp | Hộp lệch/thừa viền, sai lớp, bỏ sót |
| Instance segmentation | Một polygon + lớp cho mỗi vật, kèm `instance_id` | `kitchen`: `potted plant 0.63` gán cho lá khô treo (sai lớp); `oven 0.49` mask bị thủng | Vẽ biên theo phần nhìn thấy, mask không chồng nhau | Mask bám biên chưa, có lấn/thủng không, lớp có đúng không |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai đã cố định bằng checksum; không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ lên Colab/GitHub công khai, và không ghi họ tên, MSSV hay thông tin cá nhân vào báo cáo/output.
2. Dừng ngay, không tải lên hay commit, và báo cho Lab Coach (nếu lỡ push thì báo thêm GV).

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
