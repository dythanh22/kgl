# Thiết kế và kế hoạch triển khai KGL Desktop

**Thiết kế đề xuất, cập nhật ngày 27/11/2026.** Dùng Python cho bộ điều phối và Tkinter/ttk cho cửa sổ khởi chạy đầu tiên trên Linux. Người dùng chọn notebook đã chuẩn bị, bấm **Chạy Qwen**, xem tiến độ rồi mở chat ở cùng địa chỉ local quen thuộc. Bản đầu chạy từ checkout bằng môi trường Python của người dùng. Mốc đầu tiên cần chứng minh tài khoản Kaggle có thể đi từ notebook đang tắt tới phiên Jupyter dùng được và giải phóng đúng phiên.

## Căn cứ chọn công nghệ

| Tiêu chí | Python + ttk | Rust + Tauri |
|---|---|---|
| Kết nối Kaggle và vận hành Qwen | Dùng SDK Python tùy chọn và gọi lại luồng KGL bằng đúng interpreter. | Có thể đóng gói KGL Python làm tiến trình phụ. Phải quản lý thêm giao tiếp, phiên bản và binary theo nền tảng. Viết lại luồng vận hành bằng Rust sẽ tạo thêm một triển khai cần kiểm thử. |
| Trải nghiệm desktop | Phù hợp cho màn hình chọn notebook, theo dõi tiến độ, mở chat và dừng phiên. Chat tiếp tục dùng trình duyệt hiện có. | Đáng xem xét nếu chat phải nằm trong cửa sổ WebView của ứng dụng, hoặc cần installer và tích hợp hệ điều hành trên nhiều nền tảng. |
| Hiệu năng và tính di động | Phần chờ chính nằm ở cấp máy, đọc GGUF, nạp model và mạng. Inference do llama.cpp trên Kaggle đảm nhiệm. | Cần đo bottleneck cục bộ trước khi chọn Rust để tăng tốc. Đổi ngôn ngữ của cửa sổ vẫn để lại việc thay cơ chế quản lý proxy trên từng hệ điều hành. |

KGL hỗ trợ Python từ 3.10, CLI lõi dùng thư viện chuẩn. Trong bản kiểm tra, myenv chạy Python 3.12.16 và nạp được Tcl 8.6.15 qua tkinter. SDK Kaggle chưa có trong myenv. Khi triển khai, kiểm tra cửa sổ ttk trên desktop đích, chọn phiên bản SDK phát hành thực sự có API cần dùng và khai báo nó ở phần phụ thuộc tùy chọn. Tkinter phụ thuộc bản phân phối Python. [Manifest KGL](../pyproject.toml) · [Tkinter](https://docs.python.org/3/library/tkinter.html) · [Đóng gói sidecar trong Tauri](https://v2.tauri.app/develop/sidecar/)

## Phạm vi và giao diện bản đầu

Một tài khoản, một notebook người dùng có quyền chạy, một preset Qwen và một proxy loopback tại cổng 8001. Chuẩn bị notebook một lần với GGUF đúng preset và quyền đọc dataset bản build. Người dùng xác thực bằng cơ chế API token mà SDK hỗ trợ trên máy. Client hiển thị notebook dạng owner/slug và tình trạng xác thực.

- **Trước khi chạy** hiển thị notebook, preset, GPU yêu cầu và việc khởi động có thể dùng hạn mức Kaggle. Dùng một nút Chạy, khóa thao tác lặp trong khi đang xử lý.
- **Trong lúc chạy** hiển thị bước thực tế, thời gian chờ và Hủy. Tách chờ cấp máy, kết nối Jupyter, kiểm input, khôi phục build và nạp model để người dùng biết phần nào đang chậm.
- **Khi sẵn sàng** hiển thị model, GPU quan sát được và nút Mở chat. Mở http://127.0.0.1:8001/ trong trình duyệt mặc định. Lịch sử cũ hiện lại khi người dùng mở bằng cùng hồ sơ đã dùng cho origin này.
- **Khi kết thúc** phân biệt Đóng cửa sổ, Dừng worker và Kết thúc phiên GPU. Khi đang cấp máy, nút đóng cửa sổ cho hai lựa chọn: tiếp tục chờ trong cửa sổ hiện tại, hoặc hủy và theo dõi tới khi xác nhận được kết quả. Nếu tiến trình bị hệ điều hành đóng, lần mở sau đi qua luồng phục hồi. Khi chat đang chạy, cho phép đóng launcher và kết nối lại lúc mở sau.
- **Khi lỗi** hiển thị bước lỗi và hành động phù hợp như đổi thông tin xác thực, gắn input, chờ quota hoặc nhập URL Jupyter qua luồng KGL hiện có. Chỉ hiện nút thử lại bước có thể lặp an toàn.

## Không làm trong bản đầu

- Quản lý nhiều tài khoản, chạy nhiều notebook song song, tự động tạo notebook hoặc biên dịch lại llama.cpp cho mỗi VM.
- Scrape giao diện Kaggle, lấy cookie trình duyệt, giữ VM sống để tránh idle timeout hoặc tự cấp lại GPU sau khi phiên hết hạn.
- Đóng gói installer đa nền tảng, nhúng chat bằng WebView hoặc triển khai OAuth cho nhiều người dùng. Bản đầu phục vụ chủ tài khoản trên máy cá nhân.

## Ranh giới mã và dữ liệu

| Phần | Trách nhiệm và hợp đồng |
|---|---|
| Tích hợp Kaggle | Nhận notebook và yêu cầu GPU, gọi SDK, trả operation ID, session ID và thông tin kết nối khi có. Chuẩn hóa lỗi xác thực, quota, hết thời gian và kết quả thiếu URL. Bao riêng phần phụ thuộc vào schema của SDK. |
| Bộ điều phối Python | Là nơi duy nhất quyết định tạo, dùng lại, hủy hay phục hồi. Nhận yêu cầu chạy/dừng và phát sự kiện gồm bước, trạng thái và thông báo đã lọc. CLI thử nghiệm và cửa sổ desktop gọi chung luồng này. |
| KGL và ví dụ Qwen | Giao URL cho luồng connect qua stdin. CLI và launcher dùng chung hàm chuẩn bị Qwen, quản lý proxy và preset trong checkout. UI nhận sự kiện có cấu trúc từ các hàm này. Điều kiện sẵn sàng dựa trên health đúng instance. |
| Cửa sổ ttk | Nhận input, hiển thị sự kiện và chuyển thao tác tới bộ điều phối. Chạy công việc mạng/tiến trình ở luồng nền, cập nhật widget trên luồng UI bằng queue và after. Nút Hủy chuyển yêu cầu tới tiến trình đang sở hữu thao tác. |

KGL đã có kiểm tra phiên khi ghi session, xác minh checksum, kiểm đúng cặp GPU và kiểm chủ sở hữu cổng trước khi bàn giao proxy. Các luồng đó là nguồn xử lý dùng chung. Bản wheel hiện khai báo package data cho snippets, còn script Qwen và tài sản chat ở examples. Mốc phát hành độc lập phải có chiến lược đóng gói và bài test chạy ngoài thư mục repo. [SessionStore](../src/kgl/core/state.py) · [Chuẩn bị Qwen](../examples/qwen35b-llama/ready.py) · [Quản lý proxy](../examples/qwen35b-llama/proxy.py) · [Cấu hình gói](../pyproject.toml)

### Bố trí mã đề xuất cho M1–M3

- Đặt adapter SDK và bộ điều phối chung tại `src/kgl/desktop/kaggle.py` và `src/kgl/desktop/launcher.py`. Bộ điều phối nhận các thao tác chuẩn bị/kiểm/dừng ứng dụng qua tham số. CLI lõi không import Tk, SDK hay ví dụ Qwen khi khởi động.
- Giữ phần ghép riêng cho Qwen cạnh preset. Hai entry point dự kiến từ checkout là `python examples/qwen35b-llama/launch.py` cho CLI và `python examples/qwen35b-llama/desktop.py` cho ttk. Cả hai dùng cùng bộ điều phối và cùng hàm Qwen. Thư mục có dấu gạch nối là thư mục script; `from ready import PRESET` hiện tìm module cùng thư mục khi chạy script theo đường dẫn. Tài sản được định vị từ `__file__` để hỗ trợ gọi script từ thư mục làm việc khác.
- Extra đề xuất là `pip install -e '.[desktop]'` trong myenv. Distribution và tên import của adapter đều là `kagglesdk`; PyPI ghi bản 0.1.38 yêu cầu Python ≥3.11. M0 kiểm wheel và ghim bản tương thích trước khi thêm extra. Lõi vẫn hỗ trợ Python ≥3.10 và `dependencies = []`. `kagglehub` phục vụ khôi phục dataset trên VM. Ở M1, sửa thông báo quota của CLI lõi thành “CLI chưa truy vấn quota”, giữ khóa JSON hiện có. [PyPI kagglesdk](https://pypi.org/project/kagglesdk/0.1.38/) · [CLI status](../src/kgl/cli/commands/session.py)

### Hợp đồng tiến độ và lỗi của Qwen

Hiện `ready.py` gọi ba lệnh con nối tiếp: exec chuẩn bị trên VM, serve start và serve status. Mã thoát không đủ để UI phân biệt lỗi từng pha. Thời gian khởi động phụ thuộc cache checksum, build và VM. Callback `progress` của ServeManager nhận chuỗi; `on_start` nhận kết quả khởi động gateway. Cả hai đều nằm sau bước khôi phục build. [ready.py](../examples/qwen35b-llama/ready.py) · [Chuẩn bị trên VM](../examples/qwen35b-llama/restore_worker.py) · [ServeManager](../src/kgl/client/serve_manager.py)

Trong M2, tách các hàm kiểm GGUF/GPU, chuẩn bị hoặc xác minh build, khởi động worker và xác minh health. CLI cũ giữ cách gọi và mã thoát; launcher gọi các hàm chung từ luồng nền. Đưa sự kiện `phase`, `outcome` và `error_code` đã lọc ra **ngay khi bắt đầu/kết thúc** từng pha. Tách lần gọi remote kiểm input khỏi lần gọi chuẩn bị build để UI thấy tiến độ khi VM còn bận. Lỗi trả về tại nơi phát hiện, ví dụ thiếu input, sai GPU, lỗi lấy build, checksum sai, lỗi nạp hoặc health sai instance. Chỉ báo “thiếu quyền” khi dịch vụ xác nhận nguyên nhân đó.

Dùng lại đường chuẩn bị llama và ghi ServeRecord của CLI. Nếu tách hàm launch chung khỏi command handler, bảo toàn `on_start`, ghi revision, xác minh identity và cleanup khi start lỗi. UI dùng `progress` cho log đã lọc; quyết định điều phối dựa trên sự kiện có kiểu và health. Hủy giữa một thao tác remote đánh dấu đang hủy, chặn chạy pha kế tiếp và theo dõi kết quả. Dừng luồng chờ ở máy local chưa xác nhận worker hoặc VM đã dừng. [Lệnh serve start](../src/kgl/cli/commands/serve.py)

### Hợp đồng thao tác của bản đầu

Các lệnh dưới đây là giao diện dự kiến của `launch.py`. Cửa sổ ttk gọi cùng các thao tác Python. Chỉ bật tạo phiên tự động sau khi M0 đạt đường cấp và thu hồi. Chế độ gắn URL dùng được độc lập.

| Thao tác | Điều kiện và kết quả |
|---|---|
| `start --notebook owner/slug` | Đọc bản ghi và đối soát phiên trước. Nếu đúng phiên còn chạy, chuẩn bị hoặc dùng lại Qwen. Nếu đủ điều kiện cấp phiên mới, ghi ý định rồi mới gọi create. Kiểm notebook và tài sản local, quyền ghi session store và xung đột rõ ràng ở cổng 8001 trước khi xin GPU. Vẫn kiểm lại listener khi bind. |
| `attach --notebook owner/slug` | Nhận URL Jupyter qua lời nhắc ẩn hoặc stdin. Dùng KGL_HOME riêng, probe kernel rồi kiểm GPU/input. Slug người dùng nhập là nhãn cho kết nối thủ công. Gắn URL không cấp quyền cancel phiên Kaggle. |
| `status` | Hiện nguồn phiên, bước hiện tại, thời điểm kiểm gần nhất và hành động tiếp theo. Nếu chỉ có dữ liệu đã lưu, ghi rõ thời điểm; nút **Làm mới** yêu cầu một phép kiểm mới. |
| **Hủy khởi chạy** | Luồng điều phối đang giữ khóa nhận yêu cầu qua cờ hủy. Vô hiệu hóa nút Chạy trong lúc thu hồi. Nếu thao tác nằm trong tiến trình launcher khác, báo đang bận; bản đầu không cần daemon hay IPC điều khiển liên tiến trình. |
| `stop-worker` | Chỉ dừng instance Qwen do launcher khởi động và còn khớp bản ghi. Phiên GPU vẫn hoạt động. Hiển thị nút Chạy Qwen lại hoặc Kết thúc phiên theo quyền hiện có. |
| `detach` / `end-session` | Ngắt kết nối thủ công bằng detach. End-session chỉ áp dụng cho phiên tự tạo có bằng chứng sở hữu. Hiển thị notebook đích và cảnh báo nếu đang có inference trước khi người dùng xác nhận kết thúc. |

### Dữ liệu cần lưu và quy tắc ghi

| Nơi lưu | Nội dung |
|---|---|
| Cấu hình launcher | Slug notebook chuẩn hóa và preset. API token do SDK đọc bằng cơ chế xác thực trên máy. |
| `session.json` trong KGL_HOME riêng | Session Jupyter và ServeRecord theo hợp đồng hiện tại của KGL. URL Jupyter là thông tin xác thực, được truyền local qua stdin. |
| `launcher.json` cùng thư mục, schema v1 đề xuất | ID lần chạy, tài khoản, notebook, nguồn phiên, trạng thái, ý định đang thực hiện, operation/session ID khi biết. Ghi riêng instance worker và danh tính proxy cùng nguồn gốc từng tài nguyên. Chỉ lưu mã lỗi đã lọc; log, event và fixture loại bỏ token, URL xác thực và header. |

Bộ điều phối giữ khóa trong lúc thay đổi bản ghi. Ghi file tạm quyền 0600 trong cùng thư mục riêng tư, flush/fsync, thay file nguyên tử và fsync thư mục trên Linux trước khi gửi create. Nếu bước ghi ý định thất bại, báo lỗi local ngay. Khi nhận ID nhưng ghi thất bại, dừng các bước cấp tiếp, tận dụng ID còn trong bộ nhớ để đối soát hoặc thu hồi. Nếu bản ghi hỏng hoặc schema mới hơn bản launcher đang chạy, giữ file để khôi phục và đưa UI vào **Cần đối soát**.

Hai file có thể lệch nhau nếu launcher chết giữa hai lần ghi. Khi mở lại, so tài khoản/notebook, revision của SessionStore, kernel, instance và listener với dữ liệu hiện có trước khi thực hiện thao tác có tác dụng phụ. Một tài nguyên từng được ghi là do launcher tạo vẫn thuộc lần chạy đó khi reconnect và danh tính còn khớp. Kết quả “reused” đơn lẻ không đủ để nhận quyền sở hữu tài nguyên do người khác tạo.

## M0 cần xác minh điều gì?

Kiểm trên **bản SDK phát hành đã ghim** và tài khoản thử. Ghi phiên bản, ngày kiểm và kết quả cho từng hàng. Các quan sát mã nguồn dưới đây lấy từ nhánh main, có thể khác bản cài.

| Câu hỏi | Đã biết | Phép kiểm và quyết định |
|---|---|---|
| Token và cấu hình VM có dùng được trên notebook đích? | Request SDK có machine_shape và enable_internet. Cấu hình notebook có thể chứa dataset_sources/model_sources; cần đo việc kế thừa trên phiên tương tác. | So metadata notebook với các trường thực sự gửi. Ghi rõ trường nào bỏ qua, trường nào truyền tường minh. Kiểm quyền token, URL qua KGL/JupyterClient, đếm GPU bằng nvidia-smi và kiểm đúng file GGUF dưới /kaggle/input. Kiểm đường đọc dataset build ghim khi cần khôi phục. Một cấu hình thành công chỉ chứng minh tổ hợp đã thử. |
| Có tìm được các phiên tương tác đang chạy? | Trong KernelsApiClient đã rà có list_kernels và get_kernel_session_status. Danh sách notebook và thông tin trạng thái theo yêu cầu chưa đủ để đối chiếu một lần create bị mất phản hồi. | Kiểm cả các service liên quan và định danh trả về. Chỉ dùng cho phục hồi tự động khi đối chiếu duy nhất được tài khoản, phiên và thao tác của launcher. Nếu thiếu bằng chứng, yêu cầu chủ tài khoản kiểm tra trên Kaggle. |
| Create có chống cấp trùng ở máy chủ? | Kiểu ApiCreateKernelSessionRequest đã rà chưa thể hiện idempotency key. | Kiểm tài liệu của phiên bản ghim, header được hỗ trợ và chính sách retry ngầm của HTTP client. Chỉ bật retry create khi xác minh được bảo đảm tương ứng. Không chủ động gửi hai lệnh cấp GPU để dò hành vi. |
| Có đọc được quota? | SDK khai báo get_accelerator_quota_statistics. Schema có thời gian đã dùng, đã giữ chỗ và tổng hạn mức theo tuần. | Đọc bằng tài khoản thử, kiểm quyền, đơn vị, loại accelerator, trường thiếu và ý nghĩa phần giữ chỗ. Chỉ hiển thị số còn lại khi xác minh được cách tính. Nếu truy vấn thất bại, ẩn số liệu; lỗi hết quota khi create vẫn phải có thông báo. |
| Có thể kết nối và dừng đúng phiên? | Create trả Operation, cancel nhận Kaggle session ID. | Trên một notebook đang tắt, lấy operation/session ID và URL, chạy print(2), xác nhận hai T4 và GGUF. Ghi cách xác nhận trạng thái cuối của chính phiên sau cancel. Nếu thiếu URL, kiểm đường hủy trước khi kết luận kết quả. |
| Luồng sử dụng có phù hợp quy định? | Kaggle mô tả API token cho script tự động và truy cập Jupyter từ editor bên ngoài. Terms có giới hạn về scraping. | Trước ca live, đối chiếu Terms, AUP và quy định Notebook áp dụng cho việc chủ tài khoản tự khởi chạy SDK rồi truy cập Jupyter. Ghi ngày, nguồn và phạm vi. Nếu còn điểm mơ hồ ảnh hưởng đến luồng này, hỏi Kaggle trước khi mở rộng. Khi phát hành cho người khác, rà riêng OAuth/PKCE và đăng ký client. |

**Thứ tự ca live.** Trước khi cấp GPU, kiểm schema của wheel, Terms/AUP, chính sách retry và cách hủy. Khi bắt đầu phiên, ưu tiên create → poll → URL → `kgl connect --proxy-url -` → `print(2)` → GPU/input trong 5 phút đầu. Đây là điểm kiểm tiến độ, không phải thời hạn chắc chắn để Kaggle cấp máy. Lưu ID ngay khi có. Nếu còn xếp hàng, theo dõi operation cũ trong ngân sách M0 và dành thời gian thu hồi. Kiểm quota là phép đọc phụ. M0 chưa cần băm toàn bộ GGUF hay nạp model. Tên machine shape trong schema không bảo đảm số card thực cấp; ghi đúng cấu hình đã gửi và phần cứng quan sát được.

Nguồn kiểm M0 gồm [dịch vụ SDK](https://github.com/Kaggle/kaggle-sdk-python/blob/main/kagglesdk/kernels/services/kernels_api_service.py), [schema request và quota](https://github.com/Kaggle/kaggle-sdk-python/blob/main/kagglesdk/kernels/types/kernels_api_service.py), [xác thực API](https://www.kaggle.com/docs/api), [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks), [Terms of Use](https://www.kaggle.com/terms) và [Acceptable Use Policy](https://www.kaggle.com/aup).

## Vòng đời và phục hồi

~~~mermaid
stateDiagram-v2
    state "Chưa có phiên" as Idle
    state "Đang cấp máy" as Creating
    state "Kết nối Jupyter" as Connecting
    state "Kiểm input và nạp Qwen" as Loading
    state "Phiên còn chạy, worker đã dừng" as WorkerStopped
    state "Sẵn sàng" as Ready
    state "Chưa rõ kết quả" as Unknown
    state "Đang kết thúc" as Stopping
    state "Đã kết thúc" as Ended
    [*] --> Idle
    Idle --> Unknown: Mở lại với ý định create chưa rõ
    Idle --> Connecting: Gắn URL hoặc đối chiếu phiên đã lưu
    Idle --> Creating: Chạy mới
    Creating --> Idle: Hủy trước khi gửi create
    Creating --> Stopping: Hủy có ID phiên tự tạo
    Creating --> Unknown: Hủy sau khi gửi hoặc chỉ có operation ID
    Connecting --> Ready: Dùng lại worker và proxy đã xác minh
    Connecting --> Stopping: Hủy hoặc lỗi ở phiên tự tạo có ID
    Connecting --> Idle: Ngắt phiên gắn vào
    Loading --> Idle: Hủy hoặc lỗi ở phiên gắn vào
    Ready --> Idle: Ngắt phiên gắn vào
    Unknown --> Stopping: Đã xác minh ID sở hữu và yêu cầu hủy
    Ended --> Idle: Người dùng chọn chạy lại
    Creating --> Connecting: Có URL và ID
    Creating --> Unknown: Mất phản hồi
    Connecting --> Loading: Probe đạt
    Loading --> Ready: Proxy khớp worker
    Ready --> WorkerStopped: Đã xác nhận dừng đúng worker
    WorkerStopped --> Connecting: Chạy lại trên cùng phiên
    WorkerStopped --> Stopping: Kết thúc phiên tự tạo
    WorkerStopped --> Idle: Ngắt phiên gắn vào
    WorkerStopped --> Unknown: Mất liên lạc
    Unknown --> Idle: Đối soát xác nhận chưa cấp phiên
    Ready --> Unknown: Mất liên lạc
    Unknown --> Connecting: Xác minh đúng phiên còn chạy
    Unknown --> Ended: Xác nhận phiên đã hết
    Loading --> Stopping: Hủy hoặc lỗi ở phiên tự tạo
    Ready --> Stopping: Kết thúc phiên tự tạo
    Stopping --> Unknown: Chưa xác nhận được
    Stopping --> Ended: Đã xác nhận
~~~

| Sự kiện | Xử lý |
|---|---|
| Hai launcher hoặc launcher bị kill | Trên Linux, dùng flock độc quyền trên một đường dẫn ổn định dùng chung cho tài khoản/notebook trong launcher. Mở FD không kế thừa, giữ tới cuối thao tác. Dùng close_fds=True khi tạo tiến trình con và loại FD khóa khỏi pass_fds. Kiểm bằng ca kill launcher trong khi proxy con vẫn sống. File khóa có thể còn trên đĩa; kernel nhả khóa khi FD cuối cùng đóng. Không unlink file khóa để giành quyền chạy. Khóa này độc lập với khóa ghi ngắn hạn của SessionStore. |
| Crash trước khi nhận ID | Ghi ý định bền vững trước create. Lưu ID nguyên tử ngay khi nhận được. Lần mở sau giữ trạng thái chưa rõ, kể cả khi đã lấy lại được flock. Chỉ cấp phiên mới sau khi đối soát được thao tác cũ hoặc chủ tài khoản xác nhận trên Kaggle. |
| Có operation ID | Tiếp tục poll operation cũ, dùng deadline và khoảng chờ dịch vụ trả về. Hủy hoặc lỗi khởi động ở phiên tự tạo đi qua bước xác nhận kết thúc. |
| Phiên mất từ bên ngoài | Khi mở lại hoặc Jupyter mất liên lạc, đối chiếu trạng thái Kaggle của đúng session. Timeout, lỗi quyền hoặc token hết hạn giữ bản ghi để phục hồi. Khi đã xác nhận phiên kết thúc, dọn kết nối đúng revision và áp dụng chính sách proxy bên dưới, rồi cho phép người dùng bấm Chạy lại. Chỉ coi lỗi “không tìm thấy” là bằng chứng hết phiên nếu M0 xác minh ngữ nghĩa đó. |
| Phiên gắn vào hoặc cổng bị chiếm | Chỉ ngắt kết nối của launcher với phiên gắn vào. Kiểm chủ sở hữu trước khi dừng worker/proxy. Cổng của tiến trình khác được báo là xung đột. |
| Hủy hoặc kết thúc | Nếu Jupyter đã mất nhưng còn ID phiên tự tạo, dùng API quản lý Kaggle. Giữ ID khi cancel hết thời gian chờ. Chỉ báo đã giải phóng GPU khi đối chiếu được trạng thái cuối. |

### Cổng 8001 và quyền sở hữu proxy

Đóng cửa sổ khi Ready giữ worker và proxy để tab chat tiếp tục hoạt động. Khi chọn **Kết thúc phiên GPU**, launcher dừng proxy do chính nó tạo, sau khi xác minh đúng tiến trình. Tab chat có thể báo mất kết nối local cho tới lần chạy mới trên cùng origin. Proxy được dùng lại từ trước thuộc người dùng tiếp tục giữ cổng theo quy trình CLI hiện hành. Khi ngắt phiên gắn vào, chỉ giải phóng tài nguyên local do launcher sở hữu. Nếu worker do launcher tạo vẫn chạy trên VM, hiển thị rõ trước khi ngắt và cho chọn Dừng worker; worker đã có từ trước thuộc người dùng. [Vận hành proxy hiện tại](../examples/qwen35b-llama/QWEN35B-KAGGLE.md)

M2 bổ sung hàm dừng proxy dùng chung trong proxy.py. Ghi kết quả started/switched/reused cùng PID, thời điểm bắt đầu tiến trình và KGL_HOME. Khi dừng, giữ khóa proxy, mở pidfd, gọi check_owner, đối chiếu bản ghi và listener rồi mới gửi SIGTERM qua pidfd. Cổng đã trống là thành công. Nếu danh tính thay đổi hoặc chỉ có bằng chứng reuse, giữ listener và báo kết quả. Lỗi dọn proxy local và kết quả giải phóng GPU phải được ghi riêng.

### Theo dõi khi cửa sổ hiển thị

`GET /ready` dùng địa chỉ local nhưng handler có thể gọi REST tới kernel rồi thực thi kiểm health trên VM. Khi busy, handler có thể dùng health đã cache. Vì vậy, dùng một probe có mục đích khi người dùng mở lại hoặc yêu cầu làm mới; gộp các lần nhận focus gần nhau. Sau đó hiển thị thời điểm kiểm gần nhất và đếm lùi cục bộ từ approx_session_remaining_s nếu có. Giá trị bằng 0 chỉ khiến thông tin cần xác minh lại. [Handler readiness](../src/kgl/proxy/handler.py) · [ServeManager.health](../src/kgl/client/serve_manager.py)

Nếu M0 chứng minh có phép đọc trạng thái quản lý đúng session ID, kiểm thưa khi cửa sổ đang hiển thị và ngừng khi ẩn. Nếu thiếu khả năng này, UI ghi “Sẵn sàng lúc …” và cho làm mới. Phản hồi busy giữ phiên, connection_lost hoặc mất proxy chuyển sang Unknown để đối soát. Chỉ chuyển Ended khi được xác nhận. Không dùng vòng lặp gọi /ready hoặc chạy cell để giữ VM sống.

## Các mốc và ngân sách kiểm thử

Các con số dưới đây là **ngân sách dự kiến theo phút một phiên**, chưa phải số quota Kaggle tính. Đo cả thời gian từ lúc cấp đến lúc xác nhận kết thúc. Nếu chạm ngân sách, ngừng các ca tiếp theo và ưu tiên hủy, xác nhận hoặc báo phiên cần kiểm tra thủ công.

| Mốc | Ngân sách live | Sản phẩm và điều kiện đạt |
|---|---|---|
| M0 — Xác minh API | 1 phiên, mục tiêu 10–15 phút | Hoàn thành bảng M0. Ghi rõ khả năng nào cần thao tác thủ công. Lưu fixture đã khử bí mật cho create, poll, status, quota nếu có và cancel đã quan sát. M0 chỉ cần thấy GGUF, chưa cần nạp Qwen. |
| M1 — Điều phối CLI | 0 phiên GPU | Chạy test từ fixture M0. Ưu tiên hai launch đồng thời, kill giữa create và lưu ID, timeout sau khi nhận ID, mở lại khi phiên đã hết, hủy đang chờ và phân biệt phiên gắn vào. Token hỏng và hết quota chỉ cần ca phân loại lỗi cùng thông báo. Đạt khi không cấp trùng, không dừng nhầm và không báo sai đã giải phóng. |
| M2 — Qwen đầu cuối | 1 phiên, mục tiêu 20–30 phút | Từ notebook đã tắt tới Q4_K_M trên hai T4, context và alias đúng, một SSE hoàn tất. Chạy lại trong cùng phiên, kiểm đúng instance và proxy rồi xác nhận dừng. Offline kiểm sự kiện từng pha, lỗi GPU/input/build, hủy giữa pha, proxy tự tạo/dùng lại/cổng lạ và FD khóa sau khi launcher chết. |
| M3 — Desktop | 1 phiên, mục tiêu 15–20 phút | Test UI thường xuyên bằng worker giả. Ca live cuối đi từ Chạy Qwen tới chat trên Linux, đóng/mở launcher dùng lại đúng phiên rồi kết thúc GPU. Kiểm Hủy ở các thời điểm, focus lặp và phân biệt busy, mất proxy, phiên hết hạn bằng fixture trước ca live. |

**Fixture hợp đồng.** M0 ghi lại danh sách trường cho phép ngay trong bộ nhớ trước khi lưu. Giữ kiểu, cấu trúc lồng nhau, trường tùy chọn, mã trạng thái và quan hệ giữa các ID giả. Thay URL Jupyter, token, header xác thực và định danh thật bằng giá trị giả. Gắn phiên bản SDK và nguồn quan sát. Ghi rõ ca lỗi tổng hợp để phân biệt với phản hồi đã thấy. Khi nâng SDK, chạy lại bộ fixture và xác minh hợp đồng thay đổi trước khi cập nhật dữ liệu chuẩn.

### Quyết định sau nghiệm thu

| Kết quả | Phạm vi đủ bằng chứng để triển khai |
|---|---|
| Tạo được phiên, URL dùng được, đúng GPU/input và xác nhận được kết thúc | Mở luồng start tự động cho tổ hợp SDK, tài khoản và preset đã thử. Nếu chưa có phép đối soát create mất phản hồi, giữ đường phục hồi thủ công ở trạng thái Unknown. |
| Kết nối được URL nhập tay nhưng một trong các bước cấp, nhận ID hoặc xác nhận thu hồi còn thiếu | Phát triển và nghiệm thu riêng chế độ attach. UI hướng dẫn lấy URL và kết thúc phiên trong Kaggle. Tắt nút cấp tự động. |
| Có phiên chưa đối soát sau lần thử | Ghi lại bằng chứng đã lọc và hành động cần làm trên Kaggle. Không mở phiên thử mới để thay phiên cũ. |

Bàn giao mỗi mốc bằng diff có phạm vi rõ, lệnh test có thể chạy lại và kết quả đạt/trượt/chưa xác minh. M2 kiểm hồi quy các lệnh ready.py, proxy.py, connect qua stdin, lỗi cổng bị chiếm và alias của API. M3 kiểm từ cùng hồ sơ trình duyệt rằng hội thoại cũ còn mở được sau khi đổi VM. Chỉ công bố chế độ đã qua các ca tương ứng; báo riêng kết quả dọn proxy local và thu hồi GPU.

Mỗi mốc có thể review độc lập. Chỉ chuyển sang tự động cấp phiên khi M0 xác minh được cả kết nối và thu hồi. Nếu M0 chỉ đạt đường nhập URL thủ công, có thể triển khai launcher ở phạm vi đó và công bố rõ thao tác người dùng cần thực hiện.
