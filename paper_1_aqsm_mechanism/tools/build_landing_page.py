"""
Script to rebuild index.html and docs/index.html with 100% verified ground truth
matching paper_1_aqsm_mechanism/main.tex and the latest R18VD + R50VD benchmark data.
"""
import os
import shutil

def generate_html():
    html_content = r'''<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI TrustWorthy - Decomposing Failure Modes of Real-Time DETR under Encoder Adversarial Attacks</title>
    
    <!-- Favicon -->
    <link rel="icon" type="image/svg+xml" href="assets/favicon.svg">
    <link rel="alternate icon" href="favicon.svg">

    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- FontAwesome 6 -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <!-- Google Fonts: Inter, Outfit, Fira Code -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    
    <!-- MathJax for LaTeX formulas -->
    <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      }
    };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
        }
        h1, h2, h3, h4, .font-heading {
            font-family: 'Outfit', sans-serif;
        }
        code, pre, .font-mono {
            font-family: 'Fira Code', monospace;
        }
        .gradient-text {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #0f172a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(226, 232, 240, 0.8);
        }
        /* Bilingual Visibility Control */
        html[lang="en"] .lang-vi {
            display: none !important;
        }
        html[lang="vi"] .lang-en {
            display: none !important;
        }
        /* Metric Help Button */
        .metric-help-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 15px;
            height: 15px;
            border-radius: 9999px;
            background-color: #e2e8f0;
            color: #475569;
            font-size: 10px;
            font-weight: 700;
            line-height: 1;
            margin-left: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
            vertical-align: middle;
            border: none;
            outline: none;
        }
        .metric-help-btn:hover {
            background-color: #0284c7;
            color: #ffffff;
            transform: scale(1.15);
        }
        /* Custom scrollbar for tables */
        .custom-scrollbar::-webkit-scrollbar {
            height: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
    </style>
</head>
<body class="min-h-screen flex flex-col antialiased selection:bg-sky-500 selection:text-white">

    <!-- Header / Navigation -->
    <header class="sticky top-0 z-50 bg-white/95 backdrop-blur-md border-b border-slate-200/80 transition-all">
        <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
            <!-- Brand Logo -->
            <a href="#" class="flex items-center gap-2.5 font-heading font-bold text-lg text-slate-900 tracking-tight shrink-0">
                <span class="w-8 h-8 rounded-lg bg-sky-600 text-white flex items-center justify-center text-sm shadow-md shadow-sky-500/20">
                    <i class="fa-solid fa-shield-halved"></i>
                </span>
                <span>AI <span class="text-sky-600">TrustWorthy</span></span>
            </a>

            <!-- Nav Links -->
            <nav class="hidden lg:flex items-center gap-6 text-xs sm:text-sm font-semibold text-slate-600">
                <a href="#architecture" class="hover:text-sky-600 transition-colors">
                    <span class="lang-en">Architecture</span><span class="lang-vi">Kiến trúc</span>
                </a>
                <a href="#decomposition" class="hover:text-sky-600 transition-colors">
                    <span class="lang-en">2×2 Surgery</span><span class="lang-vi">Can thiệp 2×2</span>
                </a>
                <a href="#telemetry" class="hover:text-sky-600 transition-colors">
                    <span class="lang-en">Telemetry</span><span class="lang-vi">Chẩn đoán</span>
                </a>
                <a href="#hardware" class="hover:text-sky-600 transition-colors">
                    <span class="lang-en">Edge &amp; Latency</span><span class="lang-vi">Phần cứng &amp; Độ trễ</span>
                </a>
                <a href="#generalization" class="hover:text-sky-600 transition-colors">
                    <span class="lang-en">Cross-Arch &amp; Seeds</span><span class="lang-vi">Đa Kiến trúc &amp; Seed</span>
                </a>
                <a href="#defense" class="hover:text-sky-600 transition-colors">
                    <span class="lang-en">Defenses &amp; SQA</span><span class="lang-vi">Phòng thủ &amp; SQA</span>
                </a>
                <a href="#bibtex" class="hover:text-sky-600 transition-colors">
                    <span class="lang-en">Citation</span><span class="lang-vi">Trích dẫn</span>
                </a>
            </nav>

            <!-- Actions: Language Switcher & GitHub -->
            <div class="flex items-center gap-3 shrink-0">
                <!-- Language Switcher Toggle -->
                <div class="flex items-center rounded-full bg-slate-100 p-0.5 border border-slate-200 text-xs font-semibold shadow-xs">
                    <button id="btn-lang-en" onclick="setLanguage('en')" class="px-2.5 py-1 rounded-full transition-all bg-white text-slate-900 shadow-xs" title="English">EN</button>
                    <button id="btn-lang-vi" onclick="setLanguage('vi')" class="px-2.5 py-1 rounded-full transition-all text-slate-500 hover:text-slate-900" title="Tiếng Việt">VI</button>
                </div>
                <div class="h-4 w-px bg-slate-200 hidden sm:block"></div>
                <a href="https://github.com/vietanhlee/AI_TrustWorthy" target="_blank" class="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-semibold bg-slate-900 text-white hover:bg-slate-800 transition shadow-sm">
                    <i class="fa-brands fa-github text-sm"></i>
                    <span>GitHub</span>
                </a>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-grow">

        <!-- Hero Section -->
        <section class="pt-14 pb-16 px-4 sm:px-6 lg:px-8 text-center max-w-5xl mx-auto">
            <h1 class="text-3xl sm:text-4xl md:text-5xl font-extrabold text-slate-900 tracking-tight leading-[1.15] mb-8 max-w-4xl mx-auto font-heading">
                <span class="lang-en">Decomposing Failure Modes of <span class="gradient-text">Real-Time DETR</span> under Encoder Adversarial Attacks: An Empirical Diagnostic Study</span>
                <span class="lang-vi">Phân Rã Các Kênh Gây Sụp Đổ Của <span class="gradient-text">Real-Time DETR</span> Dưới Tấn Công Đối Kháng Vào Encoder: Một Nghiên Cứu Chẩn Đoán Thực Nghiệm</span>
            </h1>

            <!-- Authors & Affiliations -->
            <div class="flex flex-wrap justify-center items-center gap-y-2 gap-x-6 text-sm font-medium text-slate-700 mb-3">
                <div class="flex items-center gap-1.5">
                    <span class="font-semibold text-slate-900">Viet-Anh Le</span>
                    <span class="text-xs text-sky-600 bg-sky-100 px-1.5 py-0.5 rounded font-mono">1,*</span>
                </div>
                <div class="flex items-center gap-1.5">
                    <span>Thuy-Quynh Thi Dao</span>
                    <span class="text-xs text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded font-mono">1</span>
                </div>
                <div class="flex items-center gap-1.5">
                    <span>Trong-Khanh Nguyen</span>
                    <span class="text-xs text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded font-mono">1</span>
                </div>
            </div>

            <div class="text-xs text-slate-500 max-w-2xl mx-auto mb-8 space-y-1">
                <p>
                    <span class="lang-en"><sup>1</sup> Faculty of Information Technology 1, Posts and Telecommunications Institute of Technology (PTIT), Hanoi, Vietnam</span>
                    <span class="lang-vi"><sup>1</sup> Khoa Công nghệ Thông tin 1, Học viện Công nghệ Bưu chính Viễn thông (PTIT), Hà Nội, Việt Nam</span>
                </p>
                <p class="text-slate-400">* <span class="lang-en">Corresponding author</span><span class="lang-vi">Tác giả liên hệ</span>: <code class="text-slate-600">levietanhtrump@gmail.com</code></p>
            </div>

            <!-- Action Badges / Links -->
            <div class="flex flex-wrap justify-center gap-3 max-w-2xl mx-auto">
                <a href="https://github.com/vietanhlee/AI_TrustWorthy/raw/master/SUBMISSION_PDFS/1_Paper_AQSM_Mechanism_Elsevier_EAAI.pdf" target="_blank" class="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm bg-sky-600 text-white hover:bg-sky-700 transition shadow-sm shadow-sky-600/20">
                    <i class="fa-solid fa-file-pdf text-base"></i>
                    <span class="lang-en">Read Paper (PDF)</span>
                    <span class="lang-vi">Đọc bài báo (PDF)</span>
                </a>
                <a href="https://github.com/vietanhlee/AI_TrustWorthy" target="_blank" class="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition shadow-sm">
                    <i class="fa-brands fa-github text-base"></i>
                    <span class="lang-en">Source Code &amp; Data</span>
                    <span class="lang-vi">Mã nguồn &amp; Dữ liệu</span>
                </a>
                <a href="#decomposition" class="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition shadow-sm">
                    <i class="fa-solid fa-flask text-base text-sky-600"></i>
                    <span class="lang-en">2×2 Surgery</span>
                    <span class="lang-vi">Can thiệp 2×2</span>
                </a>
                <a href="#generalization" class="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition shadow-sm">
                    <i class="fa-solid fa-table-cells text-base text-indigo-600"></i>
                    <span class="lang-en">Tables 13 &amp; 14</span>
                    <span class="lang-vi">Bảng 13 &amp; 14</span>
                </a>
            </div>
        </section>

        <!-- Abstract Section -->
        <section id="abstract" class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 mb-14">
            <div class="bg-white rounded-3xl p-5 sm:p-6 border border-slate-200/80 shadow-sm">
                <div class="flex items-center justify-between mb-3">
                    <div class="flex items-center gap-2">
                        <span class="w-7 h-7 rounded-lg bg-sky-50 text-sky-600 flex items-center justify-center text-xs shadow-xs">
                            <i class="fa-solid fa-file-lines"></i>
                        </span>
                        <h2 class="text-lg sm:text-xl font-extrabold text-slate-900 font-heading">
                            <span class="lang-en">Executive Summary</span>
                            <span class="lang-vi">Tóm Tắt Nghiên Cứu</span>
                        </h2>
                    </div>
                    <span class="text-[11px] font-semibold text-slate-500 bg-slate-50 px-2.5 py-0.5 rounded-full border border-slate-200 font-mono">
                        <span class="lang-en">COCO val2017 &bull; R18 (N=300) &amp; R50 (N=200) &bull; 6-Arch (N=300)</span>
                        <span class="lang-vi">COCO val2017 &bull; R18 (N=300) &amp; R50 (N=200) &bull; 6-Arch (N=300)</span>
                    </span>
                </div>

                <div class="text-xs sm:text-sm text-slate-600 leading-relaxed mb-4">
                    <p class="lang-en">
                        Real-Time Detection Transformers (RT-DETR) eliminate Non-Maximum Suppression (NMS) via bipartite matching and dynamic top-$K$ query selection atop a multi-scale hybrid encoder. Because proposal selection operates directly on encoder feature pyramids, white-box encoder perturbations simultaneously corrupt memory representations and query routing. We present an empirical diagnostic study across <strong>RT-DETR-R18VD</strong> ($N = 300$, 3 decoder layers), <strong>RT-DETR-R50VD</strong> ($N = 200$, 6 decoder layers), and a unified <strong>6-architecture benchmark</strong> ($N = 300$) on COCO val2017 (native $640 \times 640$, $S = 8,400$ tokens, $\ell_\infty$ $\epsilon = 16/255$), using Adversarial Query Selection Manipulation (AQSM) alongside pixel-space controls, Jayasundara et al. encoder attention disruption, Query Vanishing, and Standard Task PGD baselines.
                    </p>
                    <p class="lang-vi">
                        Mô hình Real-Time Detection Transformer (RT-DETR) loại bỏ thuật toán Non-Maximum Suppression (NMS) bằng phép so khớp hai phía (bipartite matching) và cơ chế chọn lọc truy vấn động top-$K$ trên kim tự tháp đặc trưng đa tầng. Do tầng lựa chọn đề xuất hoạt động trực tiếp trên đặc trưng Encoder, các nhiễu loạn đối kháng white-box đồng thời làm sai lệch cả biểu diễn bộ nhớ lẫn định tuyến vị trí truy vấn. Chúng tôi thực hiện nghiên cứu chẩn đoán thực nghiệm trên <strong>RT-DETR-R18VD</strong> ($N = 300$, 3 tầng decoder), <strong>RT-DETR-R50VD</strong> ($N = 200$, 6 tầng decoder) và bộ chuẩn <strong>6 kiến trúc</strong> ($N = 300$) trên tập COCO val2017 ($640 \times 640$, $S = 8.400$ tokens, $\epsilon = 16/255$), kết hợp AQSM với các nhóm đối chứng không gian điểm ảnh, Jayasundara et al., Query Vanishing và Task PGD.
                    </p>
                </div>

                <!-- 4 Core Findings Grid -->
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs">
                    <div class="p-3 rounded-xl bg-slate-50/80 border border-slate-200/60 flex items-start gap-2.5">
                        <span class="w-5 h-5 rounded-md bg-rose-100 text-rose-600 flex items-center justify-center text-[10px] flex-shrink-0 mt-0.5 font-bold">1</span>
                        <div>
                            <strong class="text-slate-900 block font-semibold mb-0.5">
                                <span class="lang-en">Interventional &amp; Pixel-Space Failure Decomposition:</span>
                                <span class="lang-vi">Phân Rã Kênh Gây Sụp Đổ &amp; Đối Chứng Điểm Ảnh:</span>
                            </strong>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-en">AQSM collapses mAP@50 from $66.93\%$ to $4.05\%$ on R18VD ($N=300$; 5-seed $4.00 \pm 0.74\%$) and $74.45\% \to 10.15\%$ on R50VD ($N=200$; 5-seed $8.53 \pm 0.88\%$). Memory corruption accounts for $60.59\%\text{--}71.40\%$ attribution on R18VD and $68.22\%\text{--}74.92\%$ on R50VD. Crucially, pixel-space controls corroborate this without OOD swaps: Memory-Isolated Query preserves $59.07\%$ R18VD / $68.67\%$ R50VD vs. $0.00\%$ under Pure-Memory disruption.</span>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-vi">AQSM làm sụp đổ mAP@50 từ $66.93\%$ xuống $4.05\%$ trên R18VD ($N=300$; 5-seed $4.00 \pm 0.74\%$) và $74.45\% \to 10.15\%$ trên R50VD ($N=200$; 5-seed $8.53 \pm 0.88\%$). Tổn hại bộ nhớ chiếm $60.59\%\text{--}71.40\%$ trên R18VD và $68.22\%\text{--}74.92\%$ trên R50VD. Đối chứng không gian ảnh xác nhận không cần tráo tensor OOD: Memory-Isolated Query giữ $59.07\%$ / $68.67\%$ trong khi phá bộ nhớ thuần sụp đổ về $0.00\%$.</span>
                        </div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-50/80 border border-slate-200/60 flex items-start gap-2.5">
                        <span class="w-5 h-5 rounded-md bg-amber-100 text-amber-600 flex items-center justify-center text-[10px] flex-shrink-0 mt-0.5 font-bold">2</span>
                        <div>
                            <strong class="text-slate-900 block font-semibold mb-0.5">
                                <span class="lang-en">TOST Attention Invariance vs. Center-Only Misrouting:</span>
                                <span class="lang-vi">Kiểm Định TOST Chú Ý vs. Định Tuyến Tâm Tọa Độ:</span>
                            </strong>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-en">Normalized attention entropy remains statistically equivalent within $\pm 0.02$ TOST bounds ($0.8032 \to 0.7945$ R18VD, $p = 3.46 \times 10^{-5}$; $0.8150 \to 0.8087$ R50VD, $p = 6.80 \times 10^{-6}$), while reference drift ($\Delta p = 0.4672$ / $0.4459$) cuts sampling hit ($63.53\% \to 45.44\%$ / $64.77\% \to 50.24\%$). Center-only random shifts ($33.47\%$ R18VD / $46.79\%$ R50VD) reproduce 4D matched-$L_2$ collapse ($33.92\%$ / $48.43\%$).</span>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-vi">Entropy attention duy trì tương đương thống kê trong biên $\pm 0.02$ TOST ($0.8032 \to 0.7945$ R18VD, $p = 3.46 \times 10^{-5}$; $0.8150 \to 0.8087$ R50VD, $p = 6.80 \times 10^{-6}$), nhưng độ trôi tọa độ ($\Delta p = 0.4672$ / $0.4459$) cắt giảm tỷ lệ trúng ($63.53\% \to 45.44\%$ / $64.77\% \to 50.24\%$). Dịch chuyển riêng tâm ($33.47\%$ / $46.79\%$) tái lập sụp đổ 4D ($33.92\%$ / $48.43\%$).</span>
                        </div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-50/80 border border-slate-200/60 flex items-start gap-2.5">
                        <span class="w-5 h-5 rounded-md bg-sky-100 text-sky-600 flex items-center justify-center text-[10px] flex-shrink-0 mt-0.5 font-bold">3</span>
                        <div>
                            <strong class="text-slate-900 block font-semibold mb-0.5">
                                <span class="lang-en">20-Image Interleaved Latency vs. RetinaNet+NMS:</span>
                                <span class="lang-vi">Độ Trễ Xen Kẽ 20 Ảnh vs. RetinaNet+NMS:</span>
                            </strong>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-en">Timed via CUDA events across 20 interleaved scenes on Tesla T4 (FP32), RT-DETR exhibits invariant execution under Clean, AQSM, Task PGD, and 154-box sponge stress ($36.60 \to 36.37$ ms R18VD; $74.48 \to 74.63$ ms R50VD), running $3.8\times$ faster than RetinaNet-R50-FPN-v2 ($139.73 \to 140.36$ ms with capped batched NMS).</span>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-vi">Đo đạc bằng CUDA events xen kẽ ngẫu nhiên trên 20 ảnh thực trên Tesla T4 (FP32), RT-DETR duy trì độ trễ bất biến qua Clean, AQSM, Task PGD và sponge 154 hộp ($36.60 \to 36.37$ ms R18VD; $74.48 \to 74.63$ ms R50VD), nhanh gấp $3.8\times$ so với RetinaNet-R50-FPN-v2 ($139.73 \to 140.36$ ms với NMS giới hạn ứng viên).</span>
                        </div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-50/80 border border-slate-200/60 flex items-start gap-2.5">
                        <span class="w-5 h-5 rounded-md bg-indigo-100 text-indigo-600 flex items-center justify-center text-[10px] flex-shrink-0 mt-0.5 font-bold">4</span>
                        <div>
                            <strong class="text-slate-900 block font-semibold mb-0.5">
                                <span class="lang-en">Ensemble Hardening Bounds, Calibrated SQA &amp; 6-Arch Sweep:</span>
                                <span class="lang-vi">Ranh Giới Ensemble, SQA Chuẩn Hóa &amp; Tổng Quát 6 Kiến Trúc:</span>
                            </strong>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-en">Standalone ESM ($46.67\%$ TPR), SCM, and BCM suffer $84\%\text{--}99\%$ adaptive bypass, whereas their Ensemble caps joint evasion at $0.00\%\text{--}0.48\%$. Even after logit calibration and encoder sampling, zero-shot SQA remains null ($4.03\%$ R18VD / $10.27\%$ R50VD). Dynamic-selection failure generalizes across RT-DETRv2 ($4.01\%$) and Deformable DETR ($3.99\%$) on $N = 300$.</span>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-vi">Các giám sát đơn lẻ ESM ($46.67\%$ TPR), SCM, BCM bị qua mặt $84\%\text{--}99\%$, nhưng Ensemble giữ qua mặt đồng thời ở mức $0.00\%\text{--}0.48\%$. Ngay cả khi chuẩn hóa logit và lấy mẫu encoder, SQA zero-shot vẫn không phục hồi ($4.03\%$ / $10.27\%$). Lỗ hổng chọn lọc động lặp lại trên RT-DETRv2 ($4.01\%$) và Deformable DETR ($3.99\%$).</span>
                        </div>
                    </div>
                </div>

                <!-- Collapsible Full Abstract -->
                <details class="mt-3 group">
                    <summary class="text-xs text-sky-600 hover:text-sky-700 font-semibold cursor-pointer list-none flex items-center gap-1.5 transition select-none">
                        <i class="fa-solid fa-chevron-right text-[10px] group-open:rotate-90 transition-transform"></i>
                        <span class="lang-en">Read complete manuscript abstract text</span>
                        <span class="lang-vi">Đọc toàn văn tóm tắt bài báo</span>
                    </summary>
                    <div class="mt-3 pt-3 border-t border-slate-100 text-xs text-slate-600 leading-relaxed space-y-2 bg-slate-50/50 p-3.5 rounded-xl font-serif">
                        <p class="lang-en">
                            Real-Time Detection Transformers (RT-DETR) replace Non-Maximum Suppression (NMS) with bipartite matching and dynamic top-$K$ query selection over multi-scale hybrid encoder pyramids. Because candidate selection operates directly on encoder features, white-box encoder perturbations simultaneously corrupt encoder memory ($M$) and query routing ($Q$). We investigate this failure mechanism across RT-DETR-R18VD ($N = 300$), RT-DETR-R50VD ($N = 200$), and a unified six-model cross-architecture suite ($N = 300$) on COCO val2017 ($640 \times 640$, $S = 8,400$ tokens, $\ell_\infty$ $\epsilon = 16/255$).
                        </p>
                        <p class="lang-en">
                            Our diagnostic suite establishes four principal findings: (1) Under $2 \times 2$ tensor-swap patching ($N_{\mathrm{valid}} = 299$ R18VD / $200$ R50VD), memory corruption accounts for $60.59\%\text{--}71.40\%$ of degradation on R18VD and $68.22\%\text{--}74.92\%$ on R50VD, corroborated in feasible pixel space where memory-isolated query attacks retain $59.07\%$ / $68.67\%$ mAP@50 versus $0.00\%$ under pure memory disruption. (2) Normalized deformable cross-attention entropy stays invariant within $\pm 0.02$ TOST bounds ($p = 3.46 \times 10^{-5}$ R18VD; $p = 6.80 \times 10^{-6}$ R50VD) while reference drift collapses sampling hit rate and center-only random shifts reproduce matched-$L_2$ collapse. (3) Interleaved 20-scene CUDA profiling confirms invariant RT-DETR latency ($36.60 \to 36.37$ ms R18VD; $74.48 \to 74.63$ ms R50VD), $3.8\times$ faster than capped-NMS RetinaNet-R50-FPN-v2 ($139.73 \to 140.36$ ms). (4) Adaptive white-box evasion bypasses standalone ESM/SCM/BCM watchdogs ($84\%\text{--}99\%$) while an Ensemble caps joint evasion at $0.00\%\text{--}0.48\%$, logit-calibrated encoder-sampled SQA stays null ($4.03\%$ / $10.27\%$), and dynamic selection vulnerability transfers to RT-DETRv2-R18VD ($4.01\%$) and two-stage Deformable DETR ($3.99\%$).
                        </p>
                        <p class="lang-vi">
                            Real-Time Detection Transformer (RT-DETR) thay thế hậu xử lý NMS bằng bipartite matching và chọn lọc truy vấn động top-$K$ trên kim tự tháp đặc trưng của encoder. Do tầng chọn đề xuất nằm trực tiếp trên đầu ra encoder, nhiễu loạn white-box đồng thời phá hoại bộ nhớ ($M$) và định tuyến truy vấn ($Q$). Chúng tôi phân tích cơ chế sụp đổ trên RT-DETR-R18VD ($N = 300$), RT-DETR-R50VD ($N = 200$) và bộ chuẩn 6 kiến trúc ($N = 300$) trên tập COCO val2017 ($640 \times 640$, $S = 8.400$ tokens, $\epsilon = 16/255$).
                        </p>
                        <p class="lang-vi">
                            Nghiên cứu xác lập bốn phát hiện cốt lõi: (1) Dưới giao thức hoán đổi tensor $2 \times 2$, tổn hại bộ nhớ chiếm $60.59\%\text{--}71.40\%$ trên R18VD và $68.22\%\text{--}74.92\%$ trên R50VD, được xác chứng trong không gian ảnh khả thi khi tấn công cô lập query vẫn giữ được $59.07\%$ / $68.67\%$ mAP@50 so với $0.00\%$ khi phá bộ nhớ thuần. (2) Entropy attention duy trì bất biến trong biên $\pm 0.02$ TOST ($p = 3.46 \times 10^{-5}$ / $6.80 \times 10^{-6}$), trong khi dịch chuyển riêng tâm tọa độ tái lập sụp đổ $L_2$. (3) Đo đạc CUDA xen kẽ 20 ảnh cho thấy độ trễ RT-DETR ổn định ($36.60 \to 36.37$ ms R18VD; $74.48 \to 74.63$ ms R50VD), nhanh hơn $3.8\times$ so với RetinaNet-R50-FPN-v2. (4) Đối thủ thích nghi vượt qua từng giám sát đơn lẻ ($84\%\text{--}99\%$) nhưng Ensemble kìm giữ tỷ lệ qua mặt đồng thời còn $0.00\%\text{--}0.48\%$, SQA chuẩn hóa logit vẫn không phục hồi ($4.03\%$ / $10.27\%$), và lỗ hổng này lặp lại trên RT-DETRv2 ($4.01\%$) và Deformable DETR ($3.99\%$).
                        </p>
                    </div>
                </details>
            </div>
        </section>

        <!-- System Architecture & Threat Taxonomy -->
        <section id="architecture" class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 mb-20">
            <div class="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-sm space-y-12">
                
                <!-- Figure 1 -->
                <div>
                    <div class="mb-6 text-center max-w-2xl mx-auto">
                        <span class="text-xs font-bold text-sky-600 uppercase tracking-wider">
                            <span class="lang-en">System Architecture &amp; Threat Interface</span>
                            <span class="lang-vi">Kiến trúc Hệ thống &amp; Điểm Nghẽn Đe Dọa</span>
                        </span>
                        <h2 class="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1 mb-3">
                            <span class="lang-en">Figure 1: The Encoder-Decoder Dynamic Query Bottleneck</span>
                            <span class="lang-vi">Hình 1: Điểm Nghẽn Lựa Chọn Truy Vấn Động Giữa Encoder và Decoder</span>
                        </h2>
                        <p class="text-sm text-slate-600 lang-en">
                            RT-DETR evaluates $S = 8,400$ candidate anchors across multi-scale feature maps ($S_3, S_4, S_5$) and dynamically selects the top $N_q = 300$ highest-confidence proposals to initialize decoder content queries and reference coordinates.
                        </p>
                        <p class="text-sm text-slate-600 lang-vi">
                            RT-DETR đánh giá $S = 8.400$ anchor ứng viên trên các bản đồ đặc trưng đa tỉ lệ ($S_3, S_4, S_5$) và lựa chọn linh hoạt $N_q = 300$ đề xuất có độ tin cậy cao nhất để khởi tạo các query nội dung decoder và tọa độ tham chiếu ban đầu.
                        </p>
                    </div>

                    <div class="relative group rounded-2xl overflow-hidden border border-slate-200 mb-6 bg-slate-50 cursor-zoom-in" onclick="openModal(this.querySelector('img'))">
                        <img src="assets/images/fig1_system_architecture.png" alt="Figure 1: RT-DETR Architecture and AQSM Attack Pipeline" data-alt-en="Figure 1: RT-DETR Architecture and AQSM Attack Pipeline" data-alt-vi="Hình 1: Kiến trúc RT-DETR và Kênh Tấn công Thao túng Chọn lọc Truy vấn AQSM" class="w-full h-auto object-cover transition-transform duration-300 group-hover:scale-[1.015]">
                        <div class="absolute bottom-3 right-3 px-2.5 py-1 rounded-lg bg-slate-900/70 text-white text-[11px] font-medium backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1.5 pointer-events-none">
                            <i class="fa-solid fa-magnifying-glass-plus"></i>
                            <span class="lang-en">Click to expand</span>
                            <span class="lang-vi">Nhấn để phóng to</span>
                        </div>
                    </div>

                    <div class="bg-slate-50 rounded-xl p-4 text-xs text-slate-600 leading-relaxed border border-slate-100">
                        <div class="lang-en">
                            <strong>Figure 1 Walkthrough:</strong> (1) Input image ($640 \times 640$); (2) Hybrid Encoder extracting multi-scale feature maps ($S_3, S_4, S_5$, $S = 8,400$ tokens); (3) Dynamic Query Selection where genuine foreground proposals (green path) are evicted under Adversarial Query Selection Manipulation (AQSM) by corrupting encoder logits (red path); (4) Deformable Decoder where spatial reference coordinates undergo Spatial Target Misrouting into background clutter; (5) Output Heads generating predictions without heuristic NMS post-processing.
                        </div>
                        <div class="lang-vi">
                            <strong>Diễn giải Hình 1:</strong> (1) Ảnh đầu vào ($640 \times 640$); (2) Hybrid Encoder trích xuất bản đồ đặc trưng đa tỉ lệ ($S_3, S_4, S_5$, $S = 8.400$ tokens); (3) Tầng Dynamic Query Selection nơi các ứng viên foreground hợp lệ (nhánh xanh) bị trục xuất dưới tấn công AQSM do các logit của Encoder bị bóp méo (nhánh đỏ); (4) Deformable Decoder nơi các tọa độ tham chiếu bị định tuyến sai lệch hình học vào các cụm nền nhiễu; (5) Các đầu ra dự đoán sinh trực tiếp mà không cần thuật toán hậu xử lý NMS.
                        </div>
                    </div>
                </div>

                <!-- Figure 2: Taxonomy -->
                <div class="border-t border-slate-100 pt-10">
                    <div class="mb-6 text-center max-w-2xl mx-auto">
                        <span class="text-xs font-bold text-sky-600 uppercase tracking-wider">
                            <span class="lang-en">Threat Space Comparison</span>
                            <span class="lang-vi">So sánh Không gian Tấn công</span>
                        </span>
                        <h2 class="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1 mb-3">
                            <span class="lang-en">Figure 2: Taxonomy of Adversarial Threat Models</span>
                            <span class="lang-vi">Hình 2: Phân Loại Các Mô Hình Đe Dọa Đối Kháng</span>
                        </h2>
                        <p class="text-sm text-slate-600 lang-en">
                            Contrasting Conventional CNN Attacks, Output-Space DETR Attacks, and Encoder Query Selection Manipulation (AQSM). While CNN attacks target fixed anchor grids and vanilla DETR attacks manipulate static embeddings, AQSM targets the dynamic 8,400-to-300 selection head, inducing joint corruption of encoder memory representations and query routing priors.
                        </p>
                        <p class="text-sm text-slate-600 lang-vi">
                            Đối chiếu các cuộc tấn công anchor cố định trên CNN, tấn công so khớp hai phía ở đầu ra trên DETR thuần, và thao túng lựa chọn truy vấn bộ mã hóa (AQSM) trên RT-DETR. Trong khi tấn công CNN nhắm vào lưới anchor cố định và DETR thuần nhắm vào embedding truy vấn tĩnh, AQSM nhắm vào tầng chọn lọc linh hoạt từ 8.400 xuống 300 đề xuất, đồng thời làm sai lệch biểu diễn bộ nhớ và định tuyến không gian.
                        </p>
                    </div>

                    <div class="relative group rounded-2xl overflow-hidden border border-slate-200 mb-6 bg-slate-50 cursor-zoom-in" onclick="openModal(this.querySelector('img'))">
                        <img src="assets/images/fig2_threat_taxonomy.png" alt="Figure 2: Taxonomy of Adversarial Threat Models" data-alt-en="Figure 2: Taxonomy of Adversarial Threat Models" data-alt-vi="Hình 2: Phân loại các mô hình đe dọa đối kháng" class="w-full h-auto object-cover transition-transform duration-300 group-hover:scale-[1.015]">
                        <div class="absolute bottom-3 right-3 px-2.5 py-1 rounded-lg bg-slate-900/70 text-white text-[11px] font-medium backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1.5 pointer-events-none">
                            <i class="fa-solid fa-magnifying-glass-plus"></i>
                            <span class="lang-en">Click to expand</span>
                            <span class="lang-vi">Nhấn để phóng to</span>
                        </div>
                    </div>

                    <!-- 3-Way Threat Taxonomy Cards -->
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-3.5 text-xs">
                        <div class="bg-slate-50 rounded-xl p-4 border border-slate-200/60 flex flex-col justify-between">
                            <div>
                                <div class="flex items-center gap-2 mb-1.5 text-slate-800 font-bold">
                                    <span class="w-6 h-6 rounded-lg bg-slate-200 text-slate-700 flex items-center justify-center text-xs">A</span>
                                    <span class="lang-en">Conventional CNN Attacks</span>
                                    <span class="lang-vi">Tấn công CNN Truyền thống</span>
                                </div>
                                <p class="text-slate-600 leading-relaxed lang-en">
                                    Target fixed dense anchor grids ($H \times W \times A$). Adversarial gradients manipulate candidate classifications, flooding or suppressing heuristic Non-Maximum Suppression (NMS) windows.
                                </p>
                                <p class="text-slate-600 leading-relaxed lang-vi">
                                    Nhắm vào lưới anchor cố định dày đặc ($H \times W \times A$). Gradient đối kháng bóp méo phân loại ứng viên, gây ngập lụt hoặc vô hiệu hóa cửa sổ Non-Maximum Suppression (NMS).
                                </p>
                            </div>
                            <div class="mt-3 pt-2.5 border-t border-slate-200/60 font-mono text-[11px] text-slate-500">
                                <span class="lang-en">Target: Fixed Spatial Grid</span>
                                <span class="lang-vi">Mục tiêu: Lưới Anchor Cố định</span>
                            </div>
                        </div>

                        <div class="bg-slate-50 rounded-xl p-4 border border-slate-200/60 flex flex-col justify-between">
                            <div>
                                <div class="flex items-center gap-2 mb-1.5 text-slate-800 font-bold">
                                    <span class="w-6 h-6 rounded-lg bg-indigo-100 text-indigo-700 flex items-center justify-center text-xs">B</span>
                                    <span class="lang-en">Output-Space DETR Attacks</span>
                                    <span class="lang-vi">Tấn công Output-Space DETR</span>
                                </div>
                                <p class="text-slate-600 leading-relaxed lang-en">
                                    Target fixed learned object query embeddings in vanilla DETR / Deformable DETR. Attacks maximize Hungarian bipartite matching cost at final classification and bounding-box heads.
                                </p>
                                <p class="text-slate-600 leading-relaxed lang-vi">
                                    Nhắm vào các query embedding học sẵn cố định trong DETR / Deformable DETR gốc. Tấn công tối đa hóa hàm mất mát so khớp hai phía Hungarian tại các đầu ra phân loại và bounding-box cuối.
                                </p>
                            </div>
                            <div class="mt-3 pt-2.5 border-t border-slate-200/60 font-mono text-[11px] text-slate-500">
                                <span class="lang-en">Target: Learned Query Embeddings</span>
                                <span class="lang-vi">Mục tiêu: Embedding Truy vấn Học sẵn</span>
                            </div>
                        </div>

                        <div class="bg-sky-50/60 rounded-xl p-4 border border-sky-200/80 flex flex-col justify-between">
                            <div>
                                <div class="flex items-center gap-2 mb-1.5 text-sky-950 font-bold">
                                    <span class="w-6 h-6 rounded-lg bg-sky-600 text-white flex items-center justify-center text-xs">C</span>
                                    <span class="lang-en">AQSM: Dynamic Query Selection</span>
                                    <span class="lang-vi">AQSM: Thao túng Chọn lọc Query Động</span>
                                </div>
                                <p class="text-sky-900 leading-relaxed lang-en">
                                    Targets the dynamic $8,400 \to 300$ candidate selection head. By perturbing encoder logits, AQSM evicts foreground anchors into background noise, jointly disrupting memory and query priors.
                                </p>
                                <p class="text-sky-900 leading-relaxed lang-vi">
                                    Nhắm trúng tầng chọn lọc đề xuất động $8.400 \to 300$. Bằng cách làm sai lệch logits của encoder, AQSM trục xuất anchor foreground vào các cụm nền nhiễu, làm sụp đổ đồng thời biểu diễn bộ nhớ và vị trí query tiên nghiệm.
                                </p>
                            </div>
                            <div class="mt-3 pt-2.5 border-t border-sky-200/60 font-mono text-[11px] text-sky-700 font-semibold">
                                <span class="lang-en">Target: Encoder Selection Head ($S \to N_q$)</span>
                                <span class="lang-vi">Mục tiêu: Tầng Chọn lọc Encoder ($S \to N_q$)</span>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </section>

        <!-- 2x2 Factorial Intervention Table & Causal Surgery -->
        <section id="decomposition" class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 mb-20">
            <div class="text-center max-w-2xl mx-auto mb-10">
                <span class="text-xs font-bold text-sky-600 uppercase tracking-wider">
                    <span class="lang-en">Interventional Failure Decomposition</span>
                    <span class="lang-vi">Phân Rã Thất Bại Bằng Can Thiệp Nhân Quả</span>
                </span>
                <h2 class="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1 mb-3">
                    <span class="lang-en">2×2 Factorial Activation Patching on COCO val2017</span>
                    <span class="lang-vi">Thực Nghiệm Can Thiệp Giai Thừa 2×2 Trên COCO val2017</span>
                </h2>
                <p class="text-sm text-slate-600 lang-en">
                    Surgically swapping memory ($M$) and query ($Q$) representations across clean and adversarial passes isolates and quantifies their contributions to detector collapse across RT-DETR-R18VD ($N = 300$, $N_{\mathrm{valid}} = 299$) and RT-DETR-R50VD ($N = 200$, $N_{\mathrm{valid}} = 200$) at $\epsilon = 16/255$.
                </p>
                <p class="text-sm text-slate-600 lang-vi">
                    Can thiệp hoán đổi có kiểm soát giữa biểu diễn bộ nhớ ($M$) và truy vấn ($Q$) qua các lượt chạy sạch và đối kháng giúp cô lập và định lượng chính xác từng kênh gây sụp đổ trên RT-DETR-R18VD ($N = 300$, $N_{\mathrm{valid}} = 299$) và RT-DETR-R50VD ($N = 200$, $N_{\mathrm{valid}} = 200$) tại $\epsilon = 16/255$.
                </p>
            </div>

            <!-- Table 3 Causal Surgery Matrix -->
            <div class="bg-white rounded-3xl border border-slate-200/80 shadow-sm overflow-hidden mb-8">
                <div class="p-4 bg-slate-50/80 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <h3 class="text-sm sm:text-base font-bold text-slate-900 font-heading">
                        <span class="lang-en">Table 3: Six-Arm 2×2 Factorial Tensor-Swap Activation Patching</span>
                        <span class="lang-vi">Bảng 3: Ma Trận Can Thiệp Hoán Đổi Tensor 6 Nhánh 2×2</span>
                    </h3>
                    <span class="text-[11px] font-mono text-slate-500 bg-white px-2.5 py-1 rounded border border-slate-200 w-fit">
                        COCO val2017 &bull; R18 (N=300) / R50 (N=200) &bull; ε=16/255
                    </span>
                </div>
                <div class="overflow-x-auto custom-scrollbar">
                    <table class="w-full text-left text-sm text-slate-600">
                        <thead class="bg-slate-50/80 border-b border-slate-200 text-xs font-bold text-slate-700 uppercase tracking-wider">
                            <tr>
                                <th class="py-3.5 px-5">
                                    <span class="lang-en">Architecture &amp; Intervention Arm</span>
                                    <span class="lang-vi">Kiến trúc &amp; Nhánh Can thiệp</span>
                                </th>
                                <th class="py-3.5 px-3 text-center">
                                    <span class="lang-en">Memory ($M$)</span>
                                    <span class="lang-vi">Bộ nhớ ($M$)</span>
                                    <button type="button" class="metric-help-btn" onclick="toggleMetricPopover(this, 'memory', event)" onmouseenter="showMetricPopover(this, 'memory')" onmouseleave="scheduleClosePopover()" title="Details">?</button>
                                </th>
                                <th class="py-3.5 px-3 text-center">
                                    <span class="lang-en">Query Prior ($Q$)</span>
                                    <span class="lang-vi">Query Prior ($Q$)</span>
                                    <button type="button" class="metric-help-btn" onclick="toggleMetricPopover(this, 'query', event)" onmouseenter="showMetricPopover(this, 'query')" onmouseleave="scheduleClosePopover()" title="Details">?</button>
                                </th>
                                <th class="py-3.5 px-4 text-right">
                                    <span class="lang-en">COCO mAP@50</span>
                                    <span class="lang-vi">mAP@50 (COCO)</span>
                                    <button type="button" class="metric-help-btn" onclick="toggleMetricPopover(this, 'map50', event)" onmouseenter="showMetricPopover(this, 'map50')" onmouseleave="scheduleClosePopover()" title="Details">?</button>
                                </th>
                                <th class="py-3.5 px-4 text-right">
                                    <span class="lang-en">mAP@50:95</span>
                                    <span class="lang-vi">mAP@50:95</span>
                                    <button type="button" class="metric-help-btn" onclick="toggleMetricPopover(this, 'map5095', event)" onmouseenter="showMetricPopover(this, 'map5095')" onmouseleave="scheduleClosePopover()" title="Details">?</button>
                                </th>
                                <th class="py-3.5 px-5 text-right">
                                    <span class="lang-en">Relative Retention</span>
                                    <span class="lang-vi">Tỷ lệ Duy trì %</span>
                                    <button type="button" class="metric-help-btn" onclick="toggleMetricPopover(this, 'retention', event)" onmouseenter="showMetricPopover(this, 'retention')" onmouseleave="scheduleClosePopover()" title="Details">?</button>
                                </th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100 font-medium text-xs">
                            <!-- RT-DETR-R18VD Group -->
                            <tr class="bg-slate-100/60">
                                <td colspan="6" class="py-2 px-5 font-bold text-slate-800 text-[11px] uppercase tracking-wider">
                                    RT-DETR-R18VD (3 Decoder Layers &bull; ResNet-18 Backbone &bull; 20M Params)
                                </td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition">
                                <td class="py-3 px-5 font-semibold text-slate-900 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                                    <span class="lang-en">Arm A: Clean Baseline</span>
                                    <span class="lang-vi">Nhánh A: Mốc Sạch Danh định</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-emerald-700">Clean</td>
                                <td class="py-3 px-3 text-center font-mono text-emerald-700">Clean (300)</td>
                                <td class="py-3 px-4 text-right font-bold text-slate-900 font-mono">66.93%</td>
                                <td class="py-3 px-4 text-right font-mono">50.24%</td>
                                <td class="py-3 px-5 text-right text-emerald-600 font-bold font-mono">100.0%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-rose-50/30">
                                <td class="py-3 px-5 font-semibold text-rose-950 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                                    <span class="lang-en">Arm B: Full AQSM Attack</span>
                                    <span class="lang-vi">Nhánh B: Tấn công AQSM Đồng thời</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-rose-700 font-semibold">Adversarial</td>
                                <td class="py-3 px-3 text-center font-mono text-rose-700 font-semibold">Adversarial (300)</td>
                                <td class="py-3 px-4 text-right font-bold text-rose-600 font-mono">4.05%</td>
                                <td class="py-3 px-4 text-right font-mono text-rose-600">2.10%</td>
                                <td class="py-3 px-5 text-right text-rose-600 font-bold font-mono">6.0%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition">
                                <td class="py-3 px-5 font-semibold text-slate-900 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
                                    <span class="lang-en">Arm $\text{C}_{\text{full}}$: Adv Memory + Clean Prior</span>
                                    <span class="lang-vi">Nhánh $\text{C}_{\text{full}}$: Bộ nhớ Đối kháng + Query Sạch</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-rose-700 font-semibold">Adversarial</td>
                                <td class="py-3 px-3 text-center font-mono text-emerald-700 font-semibold">Clean (300)</td>
                                <td class="py-3 px-4 text-right font-bold text-slate-900 font-mono">9.31%</td>
                                <td class="py-3 px-4 text-right font-mono">5.02%</td>
                                <td class="py-3 px-5 text-right text-slate-600 font-bold font-mono">13.9%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-slate-50/40">
                                <td class="py-3 px-5 font-semibold text-slate-700 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-slate-400"></span>
                                    <span class="lang-en">Arm $\text{C}_{\text{index}}$: Adv Memory + Clean Index</span>
                                    <span class="lang-vi">Nhánh $\text{C}_{\text{index}}$: Bộ nhớ ĐK + Chỉ số Sạch</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-rose-700 font-semibold">Adversarial</td>
                                <td class="py-3 px-3 text-center font-mono text-purple-700 font-semibold">Adv Feat @ Clean Idx</td>
                                <td class="py-3 px-4 text-right font-bold text-slate-700 font-mono">4.86%</td>
                                <td class="py-3 px-4 text-right font-mono">2.61%</td>
                                <td class="py-3 px-5 text-right text-slate-500 font-bold font-mono">7.3%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition">
                                <td class="py-3 px-5 font-semibold text-slate-900 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-sky-500"></span>
                                    <span class="lang-en">Arm $\text{D}_{\text{full}}$: Clean Memory + Adv Prior</span>
                                    <span class="lang-vi">Nhánh $\text{D}_{\text{full}}$: Bộ nhớ Sạch + Query Đối kháng</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-emerald-700 font-semibold">Clean</td>
                                <td class="py-3 px-3 text-center font-mono text-rose-700 font-semibold">Adversarial (300)</td>
                                <td class="py-3 px-4 text-right font-bold text-slate-900 font-mono">30.37%</td>
                                <td class="py-3 px-4 text-right font-mono">20.12%</td>
                                <td class="py-3 px-5 text-right text-sky-600 font-bold font-mono">45.4%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-indigo-50/30">
                                <td class="py-3 px-5 font-semibold text-indigo-950 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-indigo-500"></span>
                                    <span class="lang-en">Arm $\text{D}_{\text{index}}$: Clean Memory + Adv Index</span>
                                    <span class="lang-vi">Nhánh $\text{D}_{\text{index}}$: Bộ nhớ Sạch + Chỉ số Đối kháng</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-emerald-700 font-semibold">Clean</td>
                                <td class="py-3 px-3 text-center font-mono text-purple-700 font-semibold">Clean Feat @ Adv Idx</td>
                                <td class="py-3 px-4 text-right font-bold text-indigo-700 font-mono">39.11%</td>
                                <td class="py-3 px-4 text-right font-mono text-indigo-700">27.21%</td>
                                <td class="py-3 px-5 text-right text-indigo-700 font-bold font-mono">58.4%</td>
                            </tr>

                            <!-- RT-DETR-R50VD Group -->
                            <tr class="bg-slate-100/60">
                                <td colspan="6" class="py-2 px-5 font-bold text-slate-800 text-[11px] uppercase tracking-wider">
                                    RT-DETR-R50VD (6 Decoder Layers &bull; ResNet-50 Backbone &bull; N = 200)
                                </td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition">
                                <td class="py-3 px-5 font-semibold text-slate-900 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                                    <span class="lang-en">Arm A: Clean Baseline</span>
                                    <span class="lang-vi">Nhánh A: Mốc Sạch Danh định</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-emerald-700">Clean</td>
                                <td class="py-3 px-3 text-center font-mono text-emerald-700">Clean (300)</td>
                                <td class="py-3 px-4 text-right font-bold text-slate-900 font-mono">74.45%</td>
                                <td class="py-3 px-4 text-right font-mono">57.47%</td>
                                <td class="py-3 px-5 text-right text-emerald-600 font-bold font-mono">100.0%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-rose-50/30">
                                <td class="py-3 px-5 font-semibold text-rose-950 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                                    <span class="lang-en">Arm B: Full AQSM Attack</span>
                                    <span class="lang-vi">Nhánh B: Tấn công AQSM Đồng thời</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-rose-700 font-semibold">Adversarial</td>
                                <td class="py-3 px-3 text-center font-mono text-rose-700 font-semibold">Adversarial (300)</td>
                                <td class="py-3 px-4 text-right font-bold text-rose-600 font-mono">10.15%</td>
                                <td class="py-3 px-4 text-right font-mono text-rose-600">5.75%</td>
                                <td class="py-3 px-5 text-right text-rose-600 font-bold font-mono">13.6%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition">
                                <td class="py-3 px-5 font-semibold text-slate-900 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
                                    <span class="lang-en">Arm $\text{C}_{\text{full}}$: Adv Memory + Clean Prior</span>
                                    <span class="lang-vi">Nhánh $\text{C}_{\text{full}}$: Bộ nhớ Đối kháng + Query Sạch</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-rose-700 font-semibold">Adversarial</td>
                                <td class="py-3 px-3 text-center font-mono text-emerald-700 font-semibold">Clean (300)</td>
                                <td class="py-3 px-4 text-right font-bold text-slate-900 font-mono">15.46%</td>
                                <td class="py-3 px-4 text-right font-mono">9.56%</td>
                                <td class="py-3 px-5 text-right text-slate-600 font-bold font-mono">20.8%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-slate-50/40">
                                <td class="py-3 px-5 font-semibold text-slate-700 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-slate-400"></span>
                                    <span class="lang-en">Arm $\text{C}_{\text{index}}$: Adv Memory + Clean Index</span>
                                    <span class="lang-vi">Nhánh $\text{C}_{\text{index}}$: Bộ nhớ ĐK + Chỉ số Sạch</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-rose-700 font-semibold">Adversarial</td>
                                <td class="py-3 px-3 text-center font-mono text-purple-700 font-semibold">Adv Feat @ Clean Idx</td>
                                <td class="py-3 px-4 text-right font-bold text-slate-700 font-mono">12.20%</td>
                                <td class="py-3 px-4 text-right font-mono">7.16%</td>
                                <td class="py-3 px-5 text-right text-slate-500 font-bold font-mono">16.4%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition">
                                <td class="py-3 px-5 font-semibold text-slate-900 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-sky-500"></span>
                                    <span class="lang-en">Arm $\text{D}_{\text{full}}$: Clean Memory + Adv Prior</span>
                                    <span class="lang-vi">Nhánh $\text{D}_{\text{full}}$: Bộ nhớ Sạch + Query Đối kháng</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-emerald-700 font-semibold">Clean</td>
                                <td class="py-3 px-3 text-center font-mono text-rose-700 font-semibold">Adversarial (300)</td>
                                <td class="py-3 px-4 text-right font-bold text-slate-900 font-mono">43.98%</td>
                                <td class="py-3 px-4 text-right font-mono">34.51%</td>
                                <td class="py-3 px-5 text-right text-sky-600 font-bold font-mono">59.1%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-indigo-50/30">
                                <td class="py-3 px-5 font-semibold text-indigo-950 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-indigo-500"></span>
                                    <span class="lang-en">Arm $\text{D}_{\text{index}}$: Clean Memory + Adv Index</span>
                                    <span class="lang-vi">Nhánh $\text{D}_{\text{index}}$: Bộ nhớ Sạch + Chỉ số Đối kháng</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-emerald-700 font-semibold">Clean</td>
                                <td class="py-3 px-3 text-center font-mono text-purple-700 font-semibold">Clean Feat @ Adv Idx</td>
                                <td class="py-3 px-4 text-right font-bold text-indigo-700 font-mono">52.82%</td>
                                <td class="py-3 px-4 text-right font-mono text-indigo-700">41.40%</td>
                                <td class="py-3 px-5 text-right text-indigo-700 font-bold font-mono">70.9%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Shapley & ANOVA Attribution Cards -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                <div class="p-4 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col justify-between">
                    <div>
                        <span class="text-[10px] font-bold uppercase tracking-wider text-rose-600">
                            <span class="lang-en">Primary Driver: Memory ($M$)</span>
                            <span class="lang-vi">Động Lực Chính: Bộ Nhớ Encoder ($M$)</span>
                        </span>
                        <div class="text-2xl font-extrabold text-slate-900 font-mono mt-1">60.59% &rarr; 68.22%</div>
                        <p class="text-xs text-slate-500 mt-1.5 leading-relaxed lang-en">
                            Per-image Shapley share on $AP_i$ is <strong>60.59%</strong> on R18VD (CI $[58.33, 62.84]$%; pooled $66.75\%$; log-scale $71.40\%$), rising to <strong>68.22%</strong> on 6-layer R50VD (pooled $72.18\%$; log-scale $74.92\%$). Repeated-measures ANOVA gives $\text{ME}_M = -42.95$ pp ($\eta_p^2 = 0.784, F = 1084.1$).
                        </p>
                        <p class="text-xs text-slate-500 mt-1.5 leading-relaxed lang-vi">
                            Tỷ trọng Shapley từng ảnh trên $AP_i$ đạt <strong>60.59%</strong> trên R18VD (CI $[58.33, 62.84]$%; pooled $66.75\%$; log-scale $71.40\%$), tăng lên <strong>68.22%</strong> trên R50VD 6 tầng (pooled $72.18\%$; log-scale $74.92\%$). ANOVA đo lường lặp lại cho $\text{ME}_M = -42.95$ pp ($\eta_p^2 = 0.784, F = 1084.1$).
                        </p>
                    </div>
                </div>

                <div class="p-4 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col justify-between">
                    <div>
                        <span class="text-[10px] font-bold uppercase tracking-wider text-sky-600">
                            <span class="lang-en">Secondary Channel: Query ($Q$)</span>
                            <span class="lang-vi">Kênh Thứ Cấp: Query Prior ($Q$)</span>
                        </span>
                        <div class="text-2xl font-extrabold text-slate-900 font-mono mt-1">39.41% &rarr; 31.78%</div>
                        <p class="text-xs text-slate-500 mt-1.5 leading-relaxed lang-en">
                            Query eviction accounts for <strong>39.41%</strong> Shapley share on R18VD (pooled $33.25\%$, log-scale $28.60\%$; $\text{ME}_Q = -27.94$ pp, $\eta_p^2 = 0.709, F = 725.2$), buffering down to <strong>31.78%</strong> on 6-layer R50VD ($\text{ME}_Q = -20.27$ pp, $\eta_p^2 = 0.576$).
                        </p>
                        <p class="text-xs text-slate-500 mt-1.5 leading-relaxed lang-vi">
                            Trục xuất query chiếm <strong>39.41%</strong> tỷ trọng Shapley trên R18VD (pooled $33.25\%$, log-scale $28.60\%$; $\text{ME}_Q = -27.94$ pp, $\eta_p^2 = 0.709, F = 725.2$), giảm xuống <strong>31.78%</strong> trên R50VD 6 tầng ($\text{ME}_Q = -20.27$ pp, $\eta_p^2 = 0.576$).
                        </p>
                    </div>
                </div>

                <div class="p-4 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col justify-between">
                    <div>
                        <span class="text-[10px] font-bold uppercase tracking-wider text-indigo-600">
                            <span class="lang-en">Clean-Memory Index Resilience ($\text{Arm D}_{\text{index}}$)</span>
                            <span class="lang-vi">Sức Chống Chịu Chỉ Số Khi Bộ Nhớ Sạch</span>
                        </span>
                        <div class="text-2xl font-extrabold text-indigo-700 font-mono mt-1">39.11% &rarr; 52.82%</div>
                        <p class="text-xs text-slate-500 mt-1.5 leading-relaxed lang-en">
                            With pristine memory $M_{\mathrm{clean}}$, purely swapping top-300 indices retains <strong>58.4%</strong> ($39.11\%$ mAP@50) on R18VD, rising to <strong>70.9%</strong> ($52.82\%$ mAP@50) on 6-layer R50VD, showing deeper decoders buffer spatial misindexing.
                        </p>
                        <p class="text-xs text-slate-500 mt-1.5 leading-relaxed lang-vi">
                            Khi giữ nguyên bộ nhớ sạch $M_{\mathrm{clean}}$, chỉ riêng tráo chỉ số top-300 vẫn giữ được <strong>58.4%</strong> ($39.11\%$ mAP@50) trên R18VD và tăng lên <strong>70.9%</strong> ($52.82\%$ mAP@50) trên R50VD, cho thấy decoder sâu hơn có khả năng bù đắp chỉ số tốt hơn.
                        </p>
                    </div>
                </div>
            </div>

            <!-- Visual Figures for Section 5: Fig 3, Fig 4 -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">
                <div class="bg-white rounded-3xl p-4 border border-slate-200/80 shadow-sm flex flex-col justify-between">
                    <div>
                        <h4 class="font-bold text-slate-900 text-xs sm:text-sm font-heading mb-1">
                            <span class="lang-en">Figure 3: Proposal Confidence Score Spectrum</span>
                            <span class="lang-vi">Hình 3: Phổ Điểm Tin Cậy Đề Xuất Top-300</span>
                        </h4>
                        <p class="text-[11px] text-slate-500 mb-3 lang-en">Boundary logit $s_{(300)}$ inflates $2.90\times$ ($0.0350 \to 0.1014$), compressing foreground separation ($M(X) = 0.6749$).</p>
                        <p class="text-[11px] text-slate-500 mb-3 lang-vi">Điểm tại biên $s_{(300)}$ phình to $2.90\times$ ($0.0350 \to 0.1014$), nén chặt khoảng cách giữa foreground và nền ($M(X) = 0.6749$).</p>
                    </div>
                    <div class="relative group rounded-2xl overflow-hidden border border-slate-200 bg-slate-50 cursor-zoom-in my-auto" onclick="openModal(this.querySelector('img'))">
                        <img src="assets/images/fig3_proposal_score_spectrum.png" alt="Figure 3: Encoder proposal confidence spectrum" data-alt-en="Figure 3: Encoder proposal confidence spectrum showing 2.90x inflation of the 300th score" data-alt-vi="Hình 3: Phổ điểm tin cậy đề xuất tại Encoder cho thấy mức tăng 2.90 lần ở hạng thứ 300" class="w-full h-auto object-contain transition-transform duration-300 group-hover:scale-[1.015]">
                    </div>
                </div>

                <div class="bg-white rounded-3xl p-4 border border-slate-200/80 shadow-sm flex flex-col justify-between">
                    <div>
                        <h4 class="font-bold text-slate-900 text-xs sm:text-sm font-heading mb-1">
                            <span class="lang-en">Figure 4: 2×2 Causal Patching &amp; ANOVA</span>
                            <span class="lang-vi">Hình 4: Can Thiệp 2×2 &amp; Phân Rã ANOVA</span>
                        </h4>
                        <p class="text-[11px] text-slate-500 mb-3 lang-en">Contrasting mAP@50 across Arms A--D and repeated-measures effect sizes ($\eta_p^2 = 0.784$ Memory vs $0.709$ Query).</p>
                        <p class="text-[11px] text-slate-500 mb-3 lang-vi">Đối chiếu mAP@50 qua các nhánh A--D và quy mô ảnh hưởng ANOVA ($\eta_p^2 = 0.784$ Bộ nhớ vs $0.709$ Query).</p>
                    </div>
                    <div class="relative group rounded-2xl overflow-hidden border border-slate-200 bg-slate-50 cursor-zoom-in my-auto" onclick="openModal(this.querySelector('img'))">
                        <img src="assets/images/fig4_causal_swap_n300.png" alt="Figure 4: Interventional activation patching and ANOVA decomposition" data-alt-en="Figure 4: Interventional activation patching and ANOVA decomposition on COCO val2017 (N=300)" data-alt-vi="Hình 4: Kết quả hoán đổi can thiệp và phân rã ANOVA trên COCO val2017 (N=300)" class="w-full h-auto object-contain transition-transform duration-300 group-hover:scale-[1.015]">
                    </div>
                </div>
            </div>
        </section>

        <!-- Geometric Telemetry Section -->
        <section id="telemetry" class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 mb-20">
            <div class="text-center max-w-2xl mx-auto mb-10">
                <span class="text-xs font-bold text-sky-600 uppercase tracking-wider">
                    <span class="lang-en">Geometric Diagnostics</span>
                    <span class="lang-vi">Chẩn đoán Hình học &amp; Cơ chế Telemetry</span>
                </span>
                <h2 class="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1 mb-3">
                    <span class="lang-en">Spatial Target Misrouting vs. Attention Scattering</span>
                    <span class="lang-vi">Định Tuyến Sai Lệch Tọa Độ vs. Tán Xạ Trọng Số Chú Ý</span>
                </h2>
                <p class="text-sm text-slate-600 lang-en">
                    Fine-grained telemetry resolves prevailing misconceptions: deformable attention weights remain sharply localized ($\approx 0.80$), but reference coordinates drift into background clutter.
                </p>
                <p class="text-sm text-slate-600 lang-vi">
                    Telemetry chi tiết làm sáng tỏ hiểu lầm phổ biến: trọng số deformable attention duy trì độ tập trung sắc nét cục bộ ($\approx 0.80$), nhưng các tọa độ tham chiếu bị định tuyến lệch hình học vào các cụm nền nhiễu ngoại vi.
                </p>
            </div>

            <!-- Two Visual Panels: Figure 5 and Figure 6 -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                <!-- Figure 5 -->
                <div class="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-sm flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between mb-1.5">
                            <h4 class="font-bold text-slate-900 text-sm font-heading">
                                <span class="lang-en">Figure 5: Feature Cosine &amp; Query Overlap</span>
                                <span class="lang-vi">Hình 5: Tương Đồng Cosine Đặc Trưng &amp; Độ Trùng Lặp Query</span>
                            </h4>
                            <span class="text-[10px] font-mono text-slate-400 bg-slate-100 px-2 py-0.5 rounded">
                                COCO val2017 &bull; N=300
                            </span>
                        </div>
                        <p class="text-xs text-slate-500 mb-3 lang-en">Multi-scale cosine degradation across $S_3, S_4, S_5$ ($0.6140$ overall R18VD $N=300$ / $0.6766$ R50VD $N=200$) and query eviction.</p>
                        <p class="text-xs text-slate-500 mb-3 lang-vi">Suy giảm cosine đa tầng qua $S_3, S_4, S_5$ ($0.6140$ R18VD $N=300$ / $0.6766$ R50VD $N=200$) và sụp đổ độ trùng lặp query.</p>
                    </div>
                    <div class="relative group rounded-2xl overflow-hidden border border-slate-200 bg-slate-50 cursor-zoom-in my-auto" onclick="openModal(this.querySelector('img'))">
                        <img src="assets/images/fig5_multiscale_cosine_and_overlap.png" alt="Figure 5: Multi-scale feature cosine similarity and query overlap dynamics" data-alt-en="Figure 5: Multi-scale feature cosine similarity and query overlap dynamics on COCO val2017" data-alt-vi="Hình 5: Độ tương đồng Cosine đặc trưng đa tầng và động thái suy giảm độ trùng lặp query trên COCO val2017" class="w-full h-auto object-contain transition-transform duration-300 group-hover:scale-[1.015]">
                        <div class="absolute bottom-2.5 right-2.5 px-2 py-0.5 rounded-md bg-slate-900/70 text-white text-[10px] font-medium backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1 pointer-events-none">
                            <i class="fa-solid fa-magnifying-glass-plus"></i>
                            <span class="lang-en">Expand</span><span class="lang-vi">Phóng to</span>
                        </div>
                    </div>
                    <div class="mt-3 pt-3 border-t border-slate-100 text-[11px] text-slate-500 leading-relaxed">
                        <span class="lang-en"><strong>Key Telemetry:</strong> Top-300 proposal overlap collapses to $16.11\%$ on R18VD ($23.21\%$ on R50VD $N=200$) while $97.22\%$ ($92.10\%$) of top-15 foreground candidate slots are evicted into peripheral background noise.</span>
                        <span class="lang-vi"><strong>Bằng chứng cốt lõi:</strong> Độ trùng lặp top-300 query sụp đổ còn $16.11\%$ trên R18VD ($23.21\%$ trên R50VD $N=200$) trong khi $97.22\%$ ($92.10\%$) khe ứng viên foreground top-15 bị trục xuất vào nhiễu nền ngoại vi.</span>
                    </div>
                </div>

                <!-- Figure 6 -->
                <div class="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-sm flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between mb-1.5">
                            <h4 class="font-bold text-slate-900 text-sm font-heading">
                                <span class="lang-en">Figure 6: Deformable Attention Telemetry</span>
                                <span class="lang-vi">Hình 6: Telemetry Chú Ý Biến Dạng (Deformable Attention)</span>
                            </h4>
                            <span class="text-[10px] font-mono text-slate-400 bg-slate-100 px-2 py-0.5 rounded">
                                Layer-wise Telemetry &bull; K=4 Points
                            </span>
                        </div>
                        <p class="text-xs text-slate-500 mb-3 lang-en">(a) Normalized attention entropy $H(A)$ vs (b) Foreground spatial sampling hit rate across decoder layers.</p>
                        <p class="text-xs text-slate-500 mb-3 lang-vi">(a) Entropy trọng số attention chuẩn hóa $H(A)$ vs (b) Tỷ lệ điểm lấy mẫu trúng foreground qua các tầng decoder.</p>
                    </div>
                    <div class="relative group rounded-2xl overflow-hidden border border-slate-200 bg-slate-50 cursor-zoom-in my-auto" onclick="openModal(this.querySelector('img'))">
                        <img src="assets/images/fig6_attention_scattering_comparison.png" alt="Figure 6: Deformable cross-attention telemetry" data-alt-en="Figure 6: Deformable cross-attention telemetry (normalized attention weight entropy vs spatial sampling hit rate across decoder layers)" data-alt-vi="Hình 6: Đo đạc telemetry cơ chế cross-attention biến dạng (Entropy trọng số chuẩn hóa vs Tỷ lệ điểm lấy mẫu trúng foreground qua các tầng decoder)" class="w-full h-auto object-contain transition-transform duration-300 group-hover:scale-[1.015]">
                        <div class="absolute bottom-2.5 right-2.5 px-2 py-0.5 rounded-md bg-slate-900/70 text-white text-[10px] font-medium backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1 pointer-events-none">
                            <i class="fa-solid fa-magnifying-glass-plus"></i>
                            <span class="lang-en">Expand</span><span class="lang-vi">Phóng to</span>
                        </div>
                    </div>
                    <div class="mt-3 pt-3 border-t border-slate-100 text-[11px] text-slate-500 leading-relaxed">
                        <span class="lang-en"><strong>Key Telemetry:</strong> Normalized entropy satisfies TOST $\pm 0.02$ equivalence ($0.8032 \to 0.7945$, $p_{\mathrm{TOST}} = 3.46 \times 10^{-5}$ R18VD; $0.8150 \to 0.8087$ R50VD). Conversely, sampling hit drops $63.53\% \to 45.44\%$ on R18VD ($p = 4.85 \times 10^{-29}, d = -0.80$) and $64.77\% \to 50.24\%$ on R50VD.</span>
                        <span class="lang-vi"><strong>Bằng chứng cốt lõi:</strong> Entropy attention đạt kiểm định tương đương TOST $\pm 0.02$ ($0.8032 \to 0.7945$, $p_{\mathrm{TOST}} = 3.46 \times 10^{-5}$ R18VD; $0.8150 \to 0.8087$ R50VD). Ngược lại, tỷ lệ lấy mẫu trúng foreground giảm mạnh $63.53\% \to 45.44\%$ trên R18VD ($p = 4.85 \times 10^{-29}, d = -0.80$) và $64.77\% \to 50.24\%$ trên R50VD.</span>
                    </div>
                </div>
            </div>

            <!-- 3 Diagnostic Takeaway Cards -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                <div class="p-4 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col justify-between">
                    <div>
                        <h4 class="font-bold text-slate-900 text-sm mb-1 font-heading flex items-center gap-2">
                            <i class="fa-solid fa-check text-emerald-500"></i>
                            <span class="lang-en">Attention Weights Stay Localized (TOST)</span>
                            <span class="lang-vi">Trọng Số Attention Vẫn Tập Trung (TOST)</span>
                        </h4>
                        <p class="text-xs text-slate-500 leading-relaxed lang-en">
                            Normalized attention entropy changes minimally ($0.8032 \to 0.7945, d = -0.15$ R18VD; $0.8150 \to 0.8087, d = -0.14$ R50VD), strictly passing paired $\pm 0.02$ TOST equivalence ($p = 3.46 \times 10^{-5} / 6.80 \times 10^{-6}$). Head shifts reweight slightly toward fine scale $S_3$ ($\max_h|\Delta H_h|=0.0615, \mathrm{TV}=0.1599$) without uniform collapse.
                        </p>
                        <p class="text-xs text-slate-500 leading-relaxed lang-vi">
                            Entropy attention chuẩn hóa chỉ đổi rất nhẹ ($0.8032 \to 0.7945, d = -0.15$ R18VD; $0.8150 \to 0.8087, d = -0.14$ R50VD), đạt kiểm định tương đương TOST $\pm 0.02$ ($p = 3.46 \times 10^{-5} / 6.80 \times 10^{-6}$). Chuyển dịch giữa các đầu chỉ tập trung nhẹ hơn về tầng $S_3$ ($\max_h|\Delta H_h|=0.0615, \mathrm{TV}=0.1599$) mà không bị tán xạ đều.
                        </p>
                    </div>
                    <div class="mt-3 pt-2 border-t border-slate-100 text-[11px] font-mono text-emerald-600 font-semibold">
                        TOST (±0.02): p = 3.46e-5 R18VD &bull; 6.80e-6 R50VD
                    </div>
                </div>

                <div class="p-4 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col justify-between">
                    <div>
                        <h4 class="font-bold text-slate-900 text-sm mb-1 font-heading flex items-center gap-2">
                            <i class="fa-solid fa-crosshairs text-rose-500"></i>
                            <span class="lang-en">Spatial Center Misrouting (Not Clamping)</span>
                            <span class="lang-vi">Định Tuyến Sai Tâm Tọa Độ</span>
                        </h4>
                        <p class="text-xs text-slate-500 leading-relaxed lang-en">
                            Reference coordinates shift sharply ($\Delta p = 0.4672, \Delta_{\mathrm{Hung}} = 0.2411$ R18VD; $\Delta p = 0.4459$ R50VD), pulling sampling hit down to $45.44\%$ ($-18.1$ pp). Pure <strong>Center-Only Random Shift</strong> ($33.47\%$ R18VD / $46.79\%$ R50VD) reproduces 4D matched-$L_2$ collapse ($33.92\%$ / $48.43\%$), ruling out box clamping artifacts.
                        </p>
                        <p class="text-xs text-slate-500 leading-relaxed lang-vi">
                            Tọa độ tham chiếu trôi mạnh ($\Delta p = 0.4672, \Delta_{\mathrm{Hung}} = 0.2411$ R18VD; $\Delta p = 0.4459$ R50VD), kéo tỷ lệ lấy mẫu trúng xuống $45.44\%$ ($-18.1$ pp). Thử nghiệm <strong>Dịch chuyển Chỉ riêng Tâm</strong> ($33.47\%$ R18VD / $46.79\%$ R50VD) tái lập suy giảm 4D ($33.92\%$ / $48.43\%$), loại trừ giả tạo co bóp kích thước hộp.
                        </p>
                    </div>
                    <div class="mt-3 pt-2 border-t border-slate-100 text-[11px] font-mono text-rose-600 font-semibold">
                        <span class="lang-en">Δp = 0.4672 &bull; Hit: 45.44% (Center-Only: 33.47%)</span>
                        <span class="lang-vi">Δp = 0.4672 &bull; Trúng FG: 45.44% (Chỉ Tâm: 33.47%)</span>
                    </div>
                </div>

                <div class="p-4 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col justify-between">
                    <div>
                        <h4 class="font-bold text-slate-900 text-sm mb-1 font-heading flex items-center gap-2">
                            <i class="fa-solid fa-scale-balanced text-indigo-500"></i>
                            <span class="lang-en">TOST Equivalence to Uniform Chance</span>
                            <span class="lang-vi">Kiểm Định TOST: Tương Đương Lấy Mẫu Đều</span>
                        </h4>
                        <p class="text-xs text-slate-500 leading-relaxed lang-en">
                            Formal Two One-Sided Tests (TOST) confirm equivalence of post-AQSM anchor foreground placement ($45.37\%$) to uniform grid chance ($45.02\%$, $\Delta = +0.20$ pp, $p_{\mathrm{TOST}} = 0.0481 < 0.05$ under $\pm 2.0$ pp bound on $N=300$).
                        </p>
                        <p class="text-xs text-slate-500 leading-relaxed lang-vi">
                            Thủ tục kiểm định tương đương hai phía (TOST) khẳng định tỷ lệ anchor trúng foreground sau AQSM ($45.37\%$) tương đương về mặt thống kê với xác suất lưới đều trên khung hình ($45.02\%$, $\Delta = +0.20$ pp, $p_{\mathrm{TOST}} = 0.0481 < 0.05$ với biên $\pm 2.0$ pp trên $N=300$).
                        </p>
                    </div>
                    <div class="mt-3 pt-2 border-t border-slate-100 text-[11px] font-mono text-indigo-600 font-semibold">
                        <span class="lang-en">p_TOST = 0.0481 &lt; 0.05 (±2.0 pp bound)</span>
                        <span class="lang-vi">p_TOST = 0.0481 &lt; 0.05 (biên ±2.0 pp)</span>
                    </div>
                </div>
            </div>

            <!-- Table 7 & Table 8 Synthesis -->
            <div class="bg-white rounded-3xl border border-slate-200/80 shadow-sm overflow-hidden">
                <div class="p-4 bg-slate-50/80 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <h3 class="text-sm sm:text-base font-bold text-slate-900 font-heading">
                        <span class="lang-en">Table 7 &amp; 8: Attention Telemetry &amp; Controlled Misrouting Summary</span>
                        <span class="lang-vi">Bảng 7 &amp; 8: Tổng Hợp Telemetry Chú Ý &amp; Thử Nghiệm Định Tuyến Sai Lệch</span>
                    </h3>
                    <span class="text-[11px] font-mono text-slate-500 bg-white px-2.5 py-1 rounded border border-slate-200 w-fit">
                        RT-DETR-R18VD &bull; N=300 &bull; 8 Heads &bull; K=4 Points
                    </span>
                </div>
                <div class="overflow-x-auto custom-scrollbar">
                    <table class="w-full text-left text-xs text-slate-600">
                        <thead class="bg-slate-100/70 border-b border-slate-200 text-[11px] font-bold text-slate-700 uppercase tracking-wider">
                            <tr>
                                <th class="py-3 px-4"><span class="lang-en">Condition / Strategy</span><span class="lang-vi">Điều kiện / Chiến lược</span></th>
                                <th class="py-3 px-3 text-center"><span class="lang-en">Norm. Entropy $H_{\mathrm{all}}$</span><span class="lang-vi">Entropy Chuẩn Hóa</span></th>
                                <th class="py-3 px-3 text-center"><span class="lang-en">Sampling FG Hit</span><span class="lang-vi">Lấy Mẫu Trúng FG</span></th>
                                <th class="py-3 px-3 text-center"><span class="lang-en">Ref Drift ($\Delta p$)</span><span class="lang-vi">Độ Trôi ($\Delta p$)</span></th>
                                <th class="py-3 px-3 text-center"><span class="lang-en">Dropoff $D(X)$</span><span class="lang-vi">Tỷ Số $D(X)$</span></th>
                                <th class="py-3 px-4 text-right"><span class="lang-en">COCO mAP@50</span><span class="lang-vi">COCO mAP@50</span></th>
                                <th class="py-3 px-4"><span class="lang-en">Mechanistic Finding</span><span class="lang-vi">Phát Hiện Cơ Chế</span></th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100 font-medium text-[11px]">
                            <tr class="hover:bg-slate-50/60 transition">
                                <td class="py-2.5 px-4 font-semibold text-slate-900 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
                                    Clean Baseline (Ref.)
                                </td>
                                <td class="py-2.5 px-3 text-center font-mono">0.8032 ± 0.0461</td>
                                <td class="py-2.5 px-3 text-center font-mono font-bold text-emerald-700">63.53%</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">0.0000</td>
                                <td class="py-2.5 px-3 text-center font-mono">34.73 ± 22.80</td>
                                <td class="py-2.5 px-4 text-right font-mono font-bold text-slate-900">66.93% (100.0%)</td>
                                <td class="py-2.5 px-4 text-slate-600">Nominal sharp focal localization; steep logit decay</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition">
                                <td class="py-2.5 px-4 font-semibold text-amber-900 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-amber-500"></span>
                                    PGD Vanishing ($T=15$)
                                </td>
                                <td class="py-2.5 px-3 text-center font-mono">0.8340 ± 0.0284</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-600">51.43%</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-600">0.4264 (Hung: 0.1643)</td>
                                <td class="py-2.5 px-3 text-center font-mono">17.67 ± 12.35</td>
                                <td class="py-2.5 px-4 text-right font-mono text-rose-600 font-bold">1.32% (2.0%)</td>
                                <td class="py-2.5 px-4 text-amber-800">Proposal suppression; complete scene blackout ($0.74$ boxes)</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition">
                                <td class="py-2.5 px-4 font-semibold text-orange-950 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-orange-500"></span>
                                    Standard Task PGD ($T=15$)
                                </td>
                                <td class="py-2.5 px-3 text-center font-mono">0.7853 ± 0.0351</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-600">47.42%</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-600">0.4534 (Hung: 0.1900)</td>
                                <td class="py-2.5 px-3 text-center font-mono">14.43 ± 8.41</td>
                                <td class="py-2.5 px-4 text-right font-mono text-rose-600 font-bold">0.35% (0.5%)</td>
                                <td class="py-2.5 px-4 text-orange-800">Dense false-alarm proposal flooding ($46.43$ boxes vs clean $8.54$)</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-rose-50/40">
                                <td class="py-2.5 px-4 font-bold text-rose-950 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-rose-600"></span>
                                    AQSM Probe ($T=15$)
                                </td>
                                <td class="py-2.5 px-3 text-center font-mono font-bold text-emerald-700">0.7945 ± 0.0316</td>
                                <td class="py-2.5 px-3 text-center font-mono font-bold text-rose-700">45.44%</td>
                                <td class="py-2.5 px-3 text-center font-mono font-bold text-rose-700">0.4672 (Hung: 0.2411)</td>
                                <td class="py-2.5 px-3 text-center font-mono font-bold text-rose-700">11.21 ± 7.98</td>
                                <td class="py-2.5 px-4 text-right font-mono font-bold text-rose-600">4.05% (6.0%)</td>
                                <td class="py-2.5 px-4 text-rose-900 font-semibold">Attention entropy TOST-equivalent ($p=3.46\times 10^{-5}$); spatial misrouting</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-sky-50/30">
                                <td class="py-2.5 px-4 font-semibold text-sky-950 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-sky-500"></span>
                                    Random Reassign (5 seeds)
                                </td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">—</td>
                                <td class="py-2.5 px-3 text-center font-mono text-sky-800">68.95% (Anchor FG)</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">0.0000 (Hung.)</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">—</td>
                                <td class="py-2.5 px-4 text-right font-mono font-bold text-sky-800">63.20 ± 0.45% (94.4%)</td>
                                <td class="py-2.5 px-4 text-sky-900">Permuting anchors across queries has minor impact (target boxes still covered)</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-sky-50/30">
                                <td class="py-2.5 px-4 font-semibold text-sky-950 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-sky-500"></span>
                                    Random Shift (4D $L_2$, 5 seeds)
                                </td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">—</td>
                                <td class="py-2.5 px-3 text-center font-mono text-sky-800">46.86% (Anchor FG)</td>
                                <td class="py-2.5 px-3 text-center font-mono text-sky-800">0.1948 (Hung.)</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">—</td>
                                <td class="py-2.5 px-4 text-right font-mono font-bold text-sky-700">33.92 ± 1.38% (50.7%)</td>
                                <td class="py-2.5 px-4 text-sky-900">Matched 4D shift halves accuracy (R50VD $N=200$: $48.43 \pm 0.93\%$)</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-indigo-50/30">
                                <td class="py-2.5 px-4 font-semibold text-indigo-950 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-indigo-500"></span>
                                    Center-Only Random Shift (5 seeds)
                                </td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">—</td>
                                <td class="py-2.5 px-3 text-center font-mono text-indigo-800">48.15% (Anchor FG)</td>
                                <td class="py-2.5 px-3 text-center font-mono text-indigo-800">0.1931 (Hung.)</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">—</td>
                                <td class="py-2.5 px-4 text-right font-mono font-bold text-indigo-700">33.47 ± 0.86% (50.0%)</td>
                                <td class="py-2.5 px-4 text-indigo-900">Pure $(c_x, c_y)$ shift with intact $(w,h)$ reproduces 4D drop (R50VD: $46.79\%$)</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-sky-50/50">
                                <td class="py-2.5 px-4 font-bold text-sky-950 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-sky-600"></span>
                                    AQSM Spatial Misrouting
                                </td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">—</td>
                                <td class="py-2.5 px-3 text-center font-mono font-bold text-sky-900">47.43% (Anchor FG)</td>
                                <td class="py-2.5 px-3 text-center font-mono font-bold text-sky-900">0.2411 (Hung.)</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">—</td>
                                <td class="py-2.5 px-4 text-right font-mono font-bold text-sky-950">30.24% (45.2%)</td>
                                <td class="py-2.5 px-4 text-sky-950 font-semibold">Adversarial clustering adds $3.24\text{--}3.68$ pp drop (R50VD: $43.78\%$)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- Qualitative Detections Showcase -->
        <section class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 mb-20">
            <div class="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-sm">
                <div class="mb-6 text-center max-w-2xl mx-auto">
                    <span class="text-xs font-bold text-sky-600 uppercase tracking-wider">
                        <span class="lang-en">Visual Evidence</span>
                        <span class="lang-vi">Minh họa Trực quan</span>
                    </span>
                    <h2 class="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1 mb-3">
                        <span class="lang-en">Figure 3b (Paper): Qualitative Detection Comparisons</span>
                        <span class="lang-vi">Hình 3b (Bài Báo): So Sánh Trực Quan Kết Quả Phát Hiện Bounding Box</span>
                    </h2>
                    <p class="text-sm text-slate-600 lang-en">
                        COCO val2017 scene ID 159791 across Clean (6 boxes, max 0.96), Gaussian noise (6 boxes kept), AQSM (0 true kept, 54 spurious), Low-budget AQSM (8 boxes), Vanishing (0 boxes, complete blackout), and Task PGD (6 boxes, 0 true kept).
                    </p>
                    <p class="text-sm text-slate-600 lang-vi">
                        Bối cảnh COCO val2017 ID 159791 đối chiếu giữa Mốc sạch (6 hộp, max 0.96), Nhiễu Gaussian (giữ nguyên 6 hộp), AQSM (0 hộp đúng, 54 hộp giả), AQSM ngân sách thấp (8 hộp), Vanishing (0 hộp, triệt tiêu hoàn toàn) và Task PGD (6 hộp, 0 hộp đúng).
                    </p>
                </div>
                <div class="relative group rounded-2xl overflow-hidden border border-slate-200 mb-4 bg-slate-50 cursor-zoom-in" onclick="openModal(this.querySelector('img'))">
                    <img src="assets/images/fig11_qualitative_detections.png" alt="Qualitative Detection Comparison" data-alt-en="Qualitative Detection Comparison across Nominal Clean, Gaussian Noise, AQSM, Vanishing, and Task PGD" data-alt-vi="So sánh trực quan kết quả phát hiện giữa Mốc sạch danh định, Nhiễu Gaussian, AQSM, Vanishing và Task PGD" class="w-full h-auto transition-transform duration-300 group-hover:scale-[1.015]">
                    <div class="absolute bottom-3 right-3 px-2.5 py-1 rounded-lg bg-slate-900/70 text-white text-[11px] font-medium backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1.5 pointer-events-none">
                        <i class="fa-solid fa-magnifying-glass-plus"></i>
                        <span class="lang-en">Click to expand</span>
                        <span class="lang-vi">Nhấn để phóng to</span>
                    </div>
                </div>
            </div>
        </section>

        <!-- Matched-Budget Sweep & Operational Hardware Execution Latency -->
        <section id="hardware" class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 mb-20">
            <div class="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-sm space-y-8">
                <div class="text-center max-w-3xl mx-auto">
                    <span class="text-xs font-bold text-sky-600 uppercase tracking-wider">
                        <span class="lang-en">Perturbation Scaling &amp; Hardware Latency Profiling</span>
                        <span class="lang-vi">Mở Rộng Ngưỡng Nhiễu &amp; Đo Đạc Độ Trễ Phần Cứng Thực Tế</span>
                    </span>
                    <h2 class="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1 mb-3 font-heading">
                        <span class="lang-en">Matched-Budget Sweep &amp; Fixed-Query Sponge Invariance</span>
                        <span class="lang-vi">Khảo Sát Đồng Ngân Sách ($\epsilon$) &amp; Tính Bất Biến Bẫy Trễ Sponge</span>
                    </h2>
                    <p class="text-sm text-slate-600 lang-en">
                        Comparing Task PGD vs AQSM across $\epsilon \in \{4, 8, 16\}/255$ (Figure 7 &amp; Table 9), combined with CUDA-event PyTorch FP32 profiling on NVIDIA Tesla T4 verifying invariant decoding under proposal flooding sponge stress (Table 10).
                    </p>
                    <p class="text-sm text-slate-600 lang-vi">
                        Đối chiếu Task PGD và AQSM trên các mức nhiễu $\epsilon \in \{4, 8, 16\}/255$ (Hình 7 &amp; Bảng 9), kết hợp đo đạc CUDA-event PyTorch FP32 trên NVIDIA Tesla T4 xác nhận tính bất biến của bộ giải mã dưới bẫy trễ proposal flooding sponge (Bảng 10).
                    </p>
                </div>

                <!-- Figure 7 + Tesla T4 Stage Breakdown -->
                <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
                    <div class="lg:col-span-7 relative group rounded-2xl overflow-hidden border border-slate-200 bg-slate-50 cursor-zoom-in" onclick="openModal(this.querySelector('img'))">
                        <img src="assets/images/fig7_matched_budget_sweep.png" alt="Figure 7: Matched-budget perturbation sweep" data-alt-en="Figure 7: Matched-budget perturbation sweep across epsilon 4/255, 8/255, and 16/255 on COCO val2017" data-alt-vi="Hình 7: Đường cong suy giảm mAP khi quét đồng mức nhiễu epsilon 4/255, 8/255 và 16/255 trên COCO val2017" class="w-full h-auto object-contain transition-transform duration-300 group-hover:scale-[1.01]">
                        <div class="absolute bottom-2.5 right-2.5 px-2 py-0.5 rounded-md bg-slate-900/70 text-white text-[10px] font-medium backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1 pointer-events-none">
                            <i class="fa-solid fa-magnifying-glass-plus"></i>
                            <span class="lang-en">Figure 7: Expand</span><span class="lang-vi">Hình 7: Phóng to</span>
                        </div>
                    </div>

                    <div class="lg:col-span-5 space-y-3 text-xs">
                        <div class="p-4 rounded-2xl bg-slate-50 border border-slate-200/70">
                            <h4 class="font-bold text-slate-900 text-sm mb-2 font-heading">
                                <span class="lang-en">Matched-Budget Sweep (Table 9)</span>
                                <span class="lang-vi">Khảo Sát Quét Ngân Sách (Bảng 9)</span>
                            </h4>
                            <p class="text-slate-600 leading-relaxed lang-en mb-2">
                                <strong>RT-DETR-R18VD ($N=300$):</strong> AQSM degrades mAP@50 monotonically: $10.80\%$ ($\epsilon=4/255$), $8.93\%$ ($\epsilon=8/255$), and $4.05\%$ ($\epsilon=16/255$). Standard Task PGD yields $2.71\%, 1.18\%, 0.35\%$.
                            </p>
                            <p class="text-slate-600 leading-relaxed lang-en">
                                <strong>RT-DETR-R50VD ($N=200$):</strong> AQSM degrades to $24.10\%$ ($4/255$), $15.40\%$ ($8/255$), and $10.15\%$ ($16/255$), vs. Task PGD $10.11\%, 6.25\%, 2.49\%$.
                            </p>
                        </div>

                        <div class="p-4 rounded-2xl bg-sky-50/70 border border-sky-200/70">
                            <h4 class="font-bold text-sky-950 text-sm mb-2 font-heading">
                                <span class="lang-en">Tesla T4 Stage Profile (36.99 ms, 27.04 FPS)</span>
                                <span class="lang-vi">Phân Rã Tác Vụ Tesla T4 (36.99 ms, 27.04 FPS)</span>
                            </h4>
                            <div class="space-y-1.5 font-mono text-[11px] text-sky-900">
                                <div class="flex justify-between"><span>Backbone (ResNet-18):</span><strong>18.94 ms (51.22%)</strong></div>
                                <div class="flex justify-between"><span>Hybrid Encoder:</span><strong>9.98 ms (26.98%)</strong></div>
                                <div class="flex justify-between"><span>Dynamic Selection Head:</span><strong>1.61 ms (4.36%)</strong></div>
                                <div class="flex justify-between"><span>Deformable Decoder (3L):</span><strong>6.16 ms (16.66%)</strong></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Table 10: Latency Sponge Profiling -->
                <div class="rounded-2xl border border-slate-200 overflow-hidden">
                    <div class="p-4 bg-slate-50/80 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                        <h3 class="text-sm sm:text-base font-bold text-slate-900 font-heading">
                            <span class="lang-en">Table 10: 20-Image Interleaved Latency &amp; RetinaNet (NMS) Benchmark</span>
                            <span class="lang-vi">Bảng 10: Đo Đạc Độ Trễ Xen Kẽ 20 Ảnh &amp; So Sánh RetinaNet (NMS)</span>
                        </h3>
                        <span class="text-[11px] font-mono text-slate-500 bg-white px-2.5 py-1 rounded border border-slate-200 w-fit">
                            Tesla T4 &bull; PyTorch FP32 &bull; Interleaved 20 Scenes &bull; N_runs=100
                        </span>
                    </div>
                    <div class="overflow-x-auto custom-scrollbar">
                        <table class="w-full text-left text-xs text-slate-600">
                            <thead class="bg-slate-100/70 border-b border-slate-200 text-[11px] font-bold text-slate-700 uppercase tracking-wider">
                                <tr>
                                    <th class="py-3 px-4"><span class="lang-en">Model</span><span class="lang-vi">Mô hình</span></th>
                                    <th class="py-3 px-4"><span class="lang-en">Interleaved Condition (20 Scenes)</span><span class="lang-vi">Điều kiện Xen Kẽ (20 Ảnh)</span></th>
                                    <th class="py-3 px-4 text-right"><span class="lang-en">Mean ± Std (ms)</span><span class="lang-vi">Độ Trễ (ms)</span></th>
                                    <th class="py-3 px-4 text-right"><span class="lang-en">Median / p95 (ms)</span><span class="lang-vi">Trung vị / p95</span></th>
                                    <th class="py-3 px-4 text-right"><span class="lang-en">Throughput (FPS)</span><span class="lang-vi">Tốc độ (FPS)</span></th>
                                    <th class="py-3 px-4 text-right"><span class="lang-en">Mean Active Boxes</span><span class="lang-vi">Số Hộp Trung Bình</span></th>
                                    <th class="py-3 px-4"><span class="lang-en">Latency Invariance Status</span><span class="lang-vi">Trạng Thái Độ Trễ</span></th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-100 font-medium text-[11px]">
                                <!-- R18VD -->
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900">RT-DETR-R18VD</td>
                                    <td class="py-2.5 px-4">Interleaved: Clean Baseline</td>
                                    <td class="py-2.5 px-4 text-right font-mono">36.60 ± 0.95</td>
                                    <td class="py-2.5 px-4 text-right font-mono">36.55 / 38.00</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-slate-900">27.32 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono">7.95 boxes</td>
                                    <td class="py-2.5 px-4 text-emerald-700 font-semibold">Nominal baseline reference (2,609 MB VRAM)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900">RT-DETR-R18VD</td>
                                    <td class="py-2.5 px-4">Interleaved: AQSM ($\epsilon = 16$)</td>
                                    <td class="py-2.5 px-4 text-right font-mono">36.47 ± 0.69</td>
                                    <td class="py-2.5 px-4 text-right font-mono">36.50 / 37.57</td>
                                    <td class="py-2.5 px-4 text-right font-mono text-emerald-700 font-bold">27.42 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono">19.90 boxes</td>
                                    <td class="py-2.5 px-4 text-emerald-700 font-semibold">Invariant ($\Delta t = -0.13$ ms)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900">RT-DETR-R18VD</td>
                                    <td class="py-2.5 px-4">Interleaved: Standard Task PGD</td>
                                    <td class="py-2.5 px-4 text-right font-mono">36.47 ± 0.76</td>
                                    <td class="py-2.5 px-4 text-right font-mono">36.51 / 37.57</td>
                                    <td class="py-2.5 px-4 text-right font-mono text-emerald-700 font-bold">27.42 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono">44.80 boxes</td>
                                    <td class="py-2.5 px-4 text-emerald-700 font-semibold">Invariant ($\Delta t = -0.13$ ms)</td>
                                </tr>
                                <tr class="hover:bg-emerald-50/40 transition bg-emerald-50/20">
                                    <td class="py-2.5 px-4 font-bold text-emerald-950">RT-DETR-R18VD</td>
                                    <td class="py-2.5 px-4 font-semibold text-emerald-900">Interleaved: Proposal-Flooding Sponge</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-emerald-700">36.37 ± 0.96</td>
                                    <td class="py-2.5 px-4 text-right font-mono">36.23 / 37.49</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-emerald-700">27.50 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-rose-600">154.20 boxes</td>
                                    <td class="py-2.5 px-4 text-emerald-800 font-bold">Immune ($19.4\times$ boxes, $-0.23$ ms delta)</td>
                                </tr>

                                <!-- R50VD -->
                                <tr class="hover:bg-slate-50/60 transition bg-slate-50/30">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900">RT-DETR-R50VD</td>
                                    <td class="py-2.5 px-4">Interleaved: Clean Baseline</td>
                                    <td class="py-2.5 px-4 text-right font-mono">74.48 ± 0.93</td>
                                    <td class="py-2.5 px-4 text-right font-mono">74.49 / 76.12</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-slate-900">13.43 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono">7.60 boxes</td>
                                    <td class="py-2.5 px-4 text-emerald-700 font-semibold">Nominal baseline (2,825 MB VRAM)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-slate-50/30">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900">RT-DETR-R50VD</td>
                                    <td class="py-2.5 px-4">Interleaved: AQSM / Task PGD</td>
                                    <td class="py-2.5 px-4 text-right font-mono">74.72 ± 1.19</td>
                                    <td class="py-2.5 px-4 text-right font-mono">74.49 / 76.41</td>
                                    <td class="py-2.5 px-4 text-right font-mono text-emerald-700 font-bold">13.38 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono">15.75 / 35.25</td>
                                    <td class="py-2.5 px-4 text-emerald-700 font-semibold">Invariant ($\Delta t = +0.25$ ms)</td>
                                </tr>
                                <tr class="hover:bg-emerald-50/40 transition bg-emerald-50/20">
                                    <td class="py-2.5 px-4 font-bold text-emerald-950">RT-DETR-R50VD</td>
                                    <td class="py-2.5 px-4 font-semibold text-emerald-900">Interleaved: Proposal-Flooding Sponge</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-emerald-700">74.63 ± 1.01</td>
                                    <td class="py-2.5 px-4 text-right font-mono">74.61 / 76.17</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-emerald-700">13.40 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-rose-600">116.00 boxes</td>
                                    <td class="py-2.5 px-4 text-emerald-800 font-bold">Immune ($15.3\times$ boxes, $+0.16$ ms delta)</td>
                                </tr>

                                <!-- RetinaNet-R50-FPN-v2 -->
                                <tr class="hover:bg-slate-50/60 transition bg-slate-100/50">
                                    <td class="py-2.5 px-4 font-semibold text-slate-800">RetinaNet-R50-FPN-v2</td>
                                    <td class="py-2.5 px-4">Interleaved: Clean (GPU Batched NMS)</td>
                                    <td class="py-2.5 px-4 text-right font-mono">139.73 ± 1.21</td>
                                    <td class="py-2.5 px-4 text-right font-mono">139.73 / 141.87</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-slate-700">7.16 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono">101.3 boxes</td>
                                    <td class="py-2.5 px-4 text-slate-600">3.8× slower than R18VD (capped top-$K$ NMS)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-slate-100/50">
                                    <td class="py-2.5 px-4 font-semibold text-slate-800">RetinaNet-R50-FPN-v2</td>
                                    <td class="py-2.5 px-4">Interleaved: Proposal-Flooding Sponge</td>
                                    <td class="py-2.5 px-4 text-right font-mono">140.36 ± 1.25</td>
                                    <td class="py-2.5 px-4 text-right font-mono">140.32 / 142.62</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-slate-700">7.12 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono text-amber-700">181.9 boxes</td>
                                    <td class="py-2.5 px-4 text-slate-600">+0.45% capped GPU NMS (<code>topk_candidates=1000</code>)</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>

        <!-- Multi-Seed Stability and Cross-Architecture Comparison (Tables 13 & 14 + Fig 9) -->
        <section id="generalization" class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 mb-20">
            <div class="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-sm space-y-8">
                <div class="text-center max-w-3xl mx-auto">
                    <span class="text-xs font-bold text-sky-600 uppercase tracking-wider">
                        <span class="lang-en">Cross-Depth, Pixel Controls &amp; 6-Architecture Suite</span>
                        <span class="lang-vi">Đối Chiếu Chiều Sâu, Không Gian Điểm Ảnh &amp; 6 Kiến Trúc</span>
                    </span>
                    <h2 class="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1 mb-3 font-heading">
                        <span class="lang-en">Tables 13 &amp; 14: Cross-Depth Consolidation &amp; 6-Model Benchmark</span>
                        <span class="lang-vi">Bảng 13 &amp; 14: Tổng Hợp Đa Chiều Sâu &amp; Kiểm Chứng 6 Kiến Trúc</span>
                    </h2>
                    <p class="text-sm text-slate-600 lang-en">
                        Direct cross-depth comparison between <strong>RT-DETR-R18VD</strong> ($N = 300$) and <strong>RT-DETR-R50VD</strong> ($N = 200$), tripartite &amp; pixel-space ablation, plus our unified <strong>$N = 300$ six-architecture benchmark</strong> spanning RT-DETR, RT-DETRv2, two-stage Deformable DETR, static Vanilla DETR, and RetinaNet+NMS.
                    </p>
                    <p class="text-sm text-slate-600 lang-vi">
                        So sánh trực tiếp giữa <strong>RT-DETR-R18VD</strong> ($N = 300$) và <strong>RT-DETR-R50VD</strong> ($N = 200$), khảo sát thành phần hàm mất mát và đối chứng không gian điểm ảnh, cùng bộ chuẩn <strong>6 kiến trúc ($N = 300$)</strong> qua RT-DETR, RT-DETRv2, Deformable DETR hai pha, Vanilla DETR tĩnh và RetinaNet+NMS.
                    </p>
                </div>

                <!-- Figure 9 Showcase -->
                <div class="relative group rounded-2xl overflow-hidden border border-slate-200 bg-slate-50 cursor-zoom-in" onclick="openModal(this.querySelector('img'))">
                    <img src="assets/images/fig9_ablation_and_multimodel.png" alt="Figure 9: Component Ablation, Pixel Controls, and 6-Architecture Cross-Model Benchmark" data-alt-en="Figure 9: (a) Tripartite loss ablation, pixel-space isolation controls, and Jayasundara et al. baseline; (b) Unified N=300 six-architecture robustness comparison." data-alt-vi="Hình 9: (a) Khảo sát cắt tỉa hàm mất mát, đối chứng điểm ảnh và Jayasundara et al.; (b) So sánh 6 kiến trúc detector trên cùng tập N=300." class="w-full h-auto object-contain transition-transform duration-300 group-hover:scale-[1.01]">
                    <div class="absolute bottom-3 right-3 px-2.5 py-1 rounded-lg bg-slate-900/70 text-white text-[11px] font-medium backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1.5 pointer-events-none">
                        <i class="fa-solid fa-magnifying-glass-plus"></i>
                        <span class="lang-en">Click to expand (Figure 9: Ablation &amp; 6-Model Suite)</span>
                        <span class="lang-vi">Nhấn để phóng to (Hình 9: Cắt tỉa &amp; 6 Kiến trúc)</span>
                    </div>
                </div>

                <!-- Table 14: Unified 6-Architecture Benchmark (N=300) -->
                <div class="rounded-2xl border border-slate-200 overflow-hidden">
                    <div class="p-4 bg-indigo-50/70 border-b border-indigo-200/70 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                        <h3 class="text-sm sm:text-base font-bold text-indigo-950 font-heading">
                            <span class="lang-en">Table 14: Unified Cross-Architecture Robustness ($N = 300$ COCO val2017)</span>
                            <span class="lang-vi">Bảng 14: Kiểm Chứng Đa Kiến Trúc Trên Cùng Tập N = 300 COCO val2017</span>
                        </h3>
                        <span class="text-[11px] font-mono text-indigo-700 bg-white px-2.5 py-1 rounded border border-indigo-200 w-fit">
                            640×640 &bull; ε=16/255, T=15 &bull; Tesla T4 FP32
                        </span>
                    </div>
                    <div class="overflow-x-auto custom-scrollbar">
                        <table class="w-full text-left text-xs text-slate-600">
                            <thead class="bg-slate-100/70 border-b border-slate-200 text-[11px] font-bold text-slate-700 uppercase tracking-wider">
                                <tr>
                                    <th class="py-3 px-4"><span class="lang-en">Detector Architecture</span><span class="lang-vi">Kiến Trúc Detector</span></th>
                                    <th class="py-3 px-3 text-center">Dec. / Dyn. Sel.</th>
                                    <th class="py-3 px-3 text-right">Clean mAP@50</th>
                                    <th class="py-3 px-3 text-right">AQSM mAP@50</th>
                                    <th class="py-3 px-3 text-right">Task PGD mAP@50</th>
                                    <th class="py-3 px-3 text-center">Evict / Cos.</th>
                                    <th class="py-3 px-4 text-right">Clean Lat.</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-100 font-medium text-[11px]">
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900">RT-DETR-R18VD</td>
                                    <td class="py-2.5 px-3 text-center font-mono">3 / Yes</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold">66.93%</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-rose-600 font-bold">5.13%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">0.35%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">83.76% / 0.4507</td>
                                    <td class="py-2.5 px-4 text-right font-mono">34.25 ms</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900">RT-DETR-R50VD</td>
                                    <td class="py-2.5 px-3 text-center font-mono">6 / Yes</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold">72.95%</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-rose-600 font-bold">8.89%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">2.93%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">77.59% / 0.5773</td>
                                    <td class="py-2.5 px-4 text-right font-mono">68.00 ms</td>
                                </tr>
                                <tr class="hover:bg-rose-50/30 transition bg-rose-50/20">
                                    <td class="py-2.5 px-4 font-bold text-rose-950">RT-DETRv2-R18VD</td>
                                    <td class="py-2.5 px-3 text-center font-mono">3 / Yes</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold">67.41%</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-rose-600 font-bold">4.01%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">0.74%</td>
                                    <td class="py-2.5 px-3 text-center font-mono font-bold">88.21% / 0.4411</td>
                                    <td class="py-2.5 px-4 text-right font-mono">32.50 ms</td>
                                </tr>
                                <tr class="hover:bg-rose-50/30 transition bg-rose-50/20">
                                    <td class="py-2.5 px-4 font-bold text-rose-950">Deformable DETR (2-Stage)</td>
                                    <td class="py-2.5 px-3 text-center font-mono">6 / Yes</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold">63.10%</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-rose-600 font-bold">3.99%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">0.14%</td>
                                    <td class="py-2.5 px-3 text-center font-mono font-bold">88.11% / 0.6063</td>
                                    <td class="py-2.5 px-4 text-right font-mono">125.76 ms</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-700">Vanilla DETR-R50 (Static)</td>
                                    <td class="py-2.5 px-3 text-center font-mono">6 / No</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold">59.47%</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-slate-400">N/A (Static)</td>
                                    <td class="py-2.5 px-3 text-right font-mono">0.72%</td>
                                    <td class="py-2.5 px-3 text-center font-mono text-slate-400">— / —</td>
                                    <td class="py-2.5 px-4 text-right font-mono">40.25 ms</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-700">RetinaNet-R50-FPN-v2 (NMS)</td>
                                    <td class="py-2.5 px-3 text-center font-mono">0 / No</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold">63.99%</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-slate-400">N/A (CNN)</td>
                                    <td class="py-2.5 px-3 text-right font-mono">0.29%</td>
                                    <td class="py-2.5 px-3 text-center font-mono text-slate-500">Noise: 61.30%</td>
                                    <td class="py-2.5 px-4 text-right font-mono">135.27 ms</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Table 13: Cross-Depth Consolidation -->
                <div class="rounded-2xl border border-slate-200 overflow-hidden">
                    <div class="p-4 bg-slate-50/80 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                        <span class="text-xs sm:text-sm font-bold text-slate-900 font-heading">
                            <span class="lang-en">Table 13: Consolidated R18VD ($N = 300$) vs. R50VD ($N = 200$) &amp; 5-Seed Stability</span>
                            <span class="lang-vi">Bảng 13: Tổng Hợp Đối Xứng R18VD ($N = 300$) vs. R50VD ($N = 200$) &amp; Độ Ổn Định 5 Seed</span>
                        </span>
                        <span class="text-[11px] font-mono text-sky-700 bg-sky-50 px-2.5 py-1 rounded border border-sky-200 w-fit">
                            ResNet-18 (20M, 3 Dec., N=300) vs. ResNet-50 (42M, 6 Dec., N=200)
                        </span>
                    </div>
                    <div class="overflow-x-auto custom-scrollbar">
                        <table class="w-full text-left text-xs text-slate-600">
                            <thead class="bg-slate-100/70 border-b border-slate-200 text-[11px] font-bold text-slate-700 uppercase tracking-wider">
                                <tr>
                                    <th class="py-3 px-5"><span class="lang-en">Diagnostic Axis ($\epsilon = 16/255$ unless noted)</span><span class="lang-vi">Trục Chẩn Đoán ($\epsilon = 16/255$ trừ khi ghi rõ)</span></th>
                                    <th class="py-3 px-4 text-center">RT-DETR-R18VD (3 dec., $N=300$)</th>
                                    <th class="py-3 px-5 text-center">RT-DETR-R50VD (6 dec., $N=200$)</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-100 font-medium text-[11px]">
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Nominal Clean mAP@50 (mAP@50:95)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">66.93% (50.24%)</td>
                                    <td class="py-2.5 px-5 text-center font-mono font-bold text-slate-900">74.45% (57.47%)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-rose-50/20">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">AQSM Cached Seed 42 mAP@50 [Rel. Drop]</td>
                                    <td class="py-2.5 px-4 text-center font-mono text-rose-600 font-semibold">4.05% [−93.96%]</td>
                                    <td class="py-2.5 px-5 text-center font-mono text-rose-600 font-semibold">10.15% [−86.36%]</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-rose-50/30">
                                    <td class="py-2.5 px-5 font-bold text-rose-950">AQSM 5-Seed Fresh Mean ± Std [Rel. Drop]</td>
                                    <td class="py-2.5 px-4 text-center font-mono font-bold text-rose-700">4.00 ± 0.74% [−94.02%]</td>
                                    <td class="py-2.5 px-5 text-center font-mono font-bold text-rose-700">8.53 ± 0.88% [−88.54%]</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">AQSM 5-Seed Hungarian Drift / Anchor FG Hit</td>
                                    <td class="py-2.5 px-4 text-center font-mono">0.2345 ± 0.0037 / 47.12 ± 0.48%</td>
                                    <td class="py-2.5 px-5 text-center font-mono">0.2207 ± 0.0036 / 51.95 ± 0.71%</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">AQSM Budget Sweep ($\epsilon = 4/255, 8/255$)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">10.80%, 8.93%</td>
                                    <td class="py-2.5 px-5 text-center font-mono">24.10%, 15.40%</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">PGD Vanishing / Task PGD / Jayasundara [27]</td>
                                    <td class="py-2.5 px-4 text-center font-mono">1.32% / 0.35% / 0.01%</td>
                                    <td class="py-2.5 px-5 text-center font-mono">4.23% / 2.49% / 0.33%</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Pixel-Space Controls (Mem-Isolated / Pure-Mem)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">59.07% (cos 0.850) / 0.00% (cos 0.070)</td>
                                    <td class="py-2.5 px-5 text-center font-mono">68.67% (cos 0.858) / 0.00% (cos 0.274)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Causal Surgery: $\text{C}_{\text{full}}$ / $\text{D}_{\text{full}}$ / $\text{D}_{\text{index}}$</td>
                                    <td class="py-2.5 px-4 text-center font-mono">9.31% / 30.37% / 39.11% (58.4% ret.)</td>
                                    <td class="py-2.5 px-5 text-center font-mono">15.46% / 43.98% / 52.82% (70.9% ret.)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-sky-50/40">
                                    <td class="py-2.5 px-5 font-bold text-sky-950">Shapley Share (Per-Image $AP_i$: Mem / Query)</td>
                                    <td class="py-2.5 px-4 text-center font-mono font-bold text-sky-900">60.59% / 39.41% [pooled 66.75%]</td>
                                    <td class="py-2.5 px-5 text-center font-mono font-bold text-sky-900">68.22% / 31.78% [pooled 72.18%]</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Encoder Memory Cosine / Top-300 Overlap / Top-15 Evict</td>
                                    <td class="py-2.5 px-4 text-center font-mono">0.6140 / 16.11% / 97.22%</td>
                                    <td class="py-2.5 px-5 text-center font-mono">0.6766 / 23.21% / 92.10%</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Deformable Attention Entropy ($H_{\mathrm{all}}$, $p_{\mathrm{TOST}}$)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">0.8032 $\to$ 0.7945 ($p = 3.46 \times 10^{-5}$)</td>
                                    <td class="py-2.5 px-5 text-center font-mono">0.8150 $\to$ 0.8087 ($p = 6.80 \times 10^{-6}$)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Deformable Sampling FG Hit Rate (Clean $\to$ AQSM)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">63.53% $\to$ 45.44% ($d = -0.80$)</td>
                                    <td class="py-2.5 px-5 text-center font-mono">64.77% $\to$ 50.24% ($d = -0.70$)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Misrouting (Reassign / 4D Shift / Center-Only / AQSM)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">63.20 / 33.92 / 33.47 / 30.24%</td>
                                    <td class="py-2.5 px-5 text-center font-mono">74.16 / 48.43 / 46.79 / 43.78%</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">ESM TPR (AUROC) / Adaptive Bypass (ESM / Ens. Joint)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">46.67% (0.7743) / 95.2–97.6% (Ens: 0.48%)</td>
                                    <td class="py-2.5 px-5 text-center font-mono">25.71% (0.6735) / 89.3–92.1% (Ens: 0.00%)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Calibrated Enc.-Sampled SQA (Stationary / Adaptive)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">4.08% $\to$ 4.03% (−0.05 pp) / 2.65%</td>
                                    <td class="py-2.5 px-5 text-center font-mono">10.15% $\to$ 10.27% (+0.12 pp) / 6.66%</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Tesla T4 Latency (Query Selection Share %)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">36.99 ms (4.36%, 27.04 FPS)</td>
                                    <td class="py-2.5 px-5 text-center font-mono">75.31 ms (2.34%, 13.28 FPS)</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>

        <!-- Defensive Hardening & Adaptive Evasion Limits -->
        <section id="defense" class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 mb-20">
            <div class="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-sm space-y-8">
                
                <div class="text-center max-w-3xl mx-auto">
                    <span class="text-xs font-bold text-sky-600 uppercase tracking-wider">
                        <span class="lang-en">Defensive Evaluation &amp; Adaptive Evasion</span>
                        <span class="lang-vi">Đánh giá Phòng thủ &amp; Ranh Giới Thích Nghi</span>
                    </span>
                    <h2 class="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1 mb-3 font-heading">
                        <span class="lang-en">Telemetry Monitors (ESM, SCM, Watchdog) &amp; Zero-Shot SQA</span>
                        <span class="lang-vi">Bộ Giám Sát Telemetry (ESM, SCM, Watchdog) &amp; Đánh Giá SQA</span>
                    </h2>
                    <p class="text-sm text-slate-600 lang-en leading-relaxed">
                        Benchmarking three runtime telemetry detectors alongside white-box Carlini-Tramèr adaptive evasion (Table 11), followed by an honest analysis of zero-shot Static Query Augmentation (SQA) showing an informative empirical null recovery result (Table 12).
                    </p>
                    <p class="text-sm text-slate-600 lang-vi leading-relaxed">
                        Đánh giá ba bộ giám sát telemetry thời gian chạy cùng ranh giới thích nghi white-box Carlini-Tramèr (Bảng 11), tiếp nối bằng phân tích trung thực về kỹ thuật Tăng Cường Truy Vấn Tĩnh (SQA) zero-shot chỉ ra kết quả null mang tính chẩn đoán (Bảng 12).
                    </p>
                </div>

                <!-- 3-Panel Figure 8 Showcase -->
                <div class="relative group rounded-2xl overflow-hidden border border-slate-200 bg-slate-50 cursor-zoom-in" onclick="openModal(this.querySelector('img'))">
                    <img src="assets/images/fig8_defense_evaluation.png" alt="Figure 8: Defense Evaluation" data-alt-en="Figure 8: Telemetry Monitoring and SQA Hardening — (a) ESM Detection TPR; (b) Carlini-Tramèr Adaptive Evasion; (c) Zero-Shot SQA Null Result." data-alt-vi="Hình 8: Đánh giá Giám sát Telemetry và Gia cố SQA — (a) Tỷ lệ phát hiện của ESM; (b) Tấn công thích nghi Carlini-Tramèr; (c) Kết quả null của SQA zero-shot." class="w-full h-auto object-cover transition-transform duration-300 group-hover:scale-[1.01]">
                    <div class="absolute bottom-3 right-3 px-2.5 py-1 rounded-lg bg-slate-900/70 text-white text-[11px] font-medium backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1.5 pointer-events-none">
                        <i class="fa-solid fa-magnifying-glass-plus"></i>
                        <span class="lang-en">Click to expand (Figure 8: 3-Panel Defense Evaluation)</span>
                        <span class="lang-vi">Nhấn để phóng to (Hình 8: Đánh giá 3 Khung hình Phòng thủ)</span>
                    </div>
                </div>

                <!-- 3 Key Findings Grid Matching Subplots (a), (b), (c) -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs leading-relaxed">
                    <div class="p-4 rounded-2xl bg-slate-50 border border-slate-200/70 flex flex-col justify-between">
                        <div>
                            <div class="flex items-center gap-2 mb-1.5 font-bold text-slate-900">
                                <span class="w-5 h-5 rounded-md bg-rose-100 text-rose-600 flex items-center justify-center text-[10px] font-bold">(a)</span>
                                <span class="lang-en">Stationary Monitor Detection</span>
                                <span class="lang-vi">Năng Lực Phát Hiện Tĩnh Của Giám Sát</span>
                            </div>
                            <p class="text-slate-600 lang-en mb-2">
                                ESM flags stationary AQSM at <strong>46.67% TPR</strong> ($\text{AUROC} = 0.7743$) on R18VD ($25.71\%$ on R50VD). Two-sided BCM catches Task PGD ($88.57\%$ / $91.43\%$ one-sided) and Vanishing ($85.24\%$).
                            </p>
                            <p class="text-slate-600 lang-vi mb-2">
                                ESM phát hiện AQSM tĩnh đạt <strong>46.67% TPR</strong> ($\text{AUROC} = 0.7743$) trên R18VD ($25.71\%$ trên R50VD). Bộ giám sát BCM bắt mạnh Task PGD ($88.57\%$ / $91.43\%$) và Vanishing ($85.24\%$).
                            </p>
                        </div>
                        <span class="text-[10px] font-mono text-slate-400 bg-white px-2 py-0.5 rounded border border-slate-200 w-fit">Table 11 &bull; N_test=210 R18 / 140 R50</span>
                    </div>

                    <div class="p-4 rounded-2xl bg-slate-50 border border-slate-200/70 flex flex-col justify-between">
                        <div>
                            <div class="flex items-center gap-2 mb-1.5 font-bold text-slate-900">
                                <span class="w-5 h-5 rounded-md bg-amber-100 text-amber-600 flex items-center justify-center text-[10px] font-bold">(b)</span>
                                <span class="lang-en">Standalone Evasion vs. Ensemble Bounds</span>
                                <span class="lang-vi">Qua Mặt Đơn Lẻ vs. Tổ Hợp Ensemble</span>
                            </div>
                            <p class="text-slate-600 lang-en mb-2">
                                Adaptive white-box evasion bypasses standalone ESM (<strong>95.24%–97.62%</strong>), SCM (<strong>94.76%–95.71%</strong>; mAP $0.76\%\text{--}1.73\%$), and BCM (<strong>84.29%–92.86%</strong>), whereas attacking their Ensemble caps joint evasion at <strong>0.00%–0.48%</strong> ($9.51\%\text{--}10.00\%$ mAP).
                            </p>
                            <p class="text-slate-600 lang-vi mb-2">
                                Tấn công thích nghi qua mặt dễ dàng ESM (<strong>95.24%–97.62%</strong>), SCM (<strong>94.76%–95.71%</strong>; mAP $0.76\%\text{--}1.73\%$) và BCM (<strong>84.29%–92.86%</strong>) đơn lẻ, nhưng Ensemble giữ qua mặt đồng thời còn <strong>0.00%–0.48%</strong>.
                            </p>
                        </div>
                        <span class="text-[10px] font-mono text-slate-400 bg-white px-2 py-0.5 rounded border border-slate-200 w-fit">Adaptive Ensemble Joint: 0.00–0.48%</span>
                    </div>

                    <div class="p-4 rounded-2xl bg-sky-50/70 border border-sky-200/70 flex flex-col justify-between">
                        <div>
                            <div class="flex items-center gap-2 mb-1.5 font-bold text-sky-950">
                                <span class="w-5 h-5 rounded-md bg-sky-200 text-sky-800 flex items-center justify-center text-[10px] font-bold">(c)</span>
                                <span class="lang-en">Calibrated Enc.-Sampled SQA Null Result</span>
                                <span class="lang-vi">Kết Quả Null SQA Sau Chuẩn Hóa Logit</span>
                            </div>
                            <p class="text-sky-900 lang-en mb-2">
                                Even after inverse-sigmoid logit calibration and encoder feature sampling, zero-shot SQA remains null: $4.08\% \to 4.03\%$ ($-0.05$ pp) on R18VD and $10.15\% \to 10.27\%$ ($+0.12$ pp) on R50VD, falling to $2.65\%$ / $6.66\%$ under full held-out adaptive attacks ($N_{\mathrm{test}} = 210 / 140$).
                            </p>
                            <p class="text-sky-900 lang-vi mb-2">
                                Ngay cả sau khi chuẩn hóa tọa độ <code>logit</code> và lấy mẫu đặc trưng từ Encoder, SQA zero-shot vẫn không phục hồi: $4.08\% \to 4.03\%$ ($-0.05$ pp) trên R18VD và $10.15\% \to 10.27\%$ ($+0.12$ pp) trên R50VD, tụt về $2.65\%$ / $6.66\%$ khi bị tấn công thích nghi toàn tập kiểm thử ($N_{\mathrm{test}} = 210 / 140$).
                            </p>
                        </div>
                        <span class="text-[10px] font-mono text-sky-700 bg-sky-100/80 px-2 py-0.5 rounded border border-sky-300 w-fit">Table 12 &bull; Calibrated SQA + Enc Sample</span>
                    </div>
                </div>

                <!-- Table 12: Architectural Hardening via SQA Validation & Controls -->
                <div class="rounded-2xl border border-slate-200 overflow-hidden">
                    <div class="p-4 sm:p-5 bg-slate-50/80 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                        <div>
                            <h3 class="text-sm sm:text-base font-bold text-slate-900 font-heading">
                                <span class="lang-en">Table 12: Calibrated &amp; Encoder-Sampled SQA Ablation &amp; Full Adaptive Evasion</span>
                                <span class="lang-vi">Bảng 12: Thử Nghiệm Chuẩn Hóa SQA &amp; Tấn Công Thích Nghi Toàn Tập</span>
                            </h3>
                            <p class="text-xs text-slate-500 mt-0.5 lang-en">COCO val2017 across RT-DETR-R18VD ($N = 300$, $N_{\mathrm{test}} = 210$) and RT-DETR-R50VD ($N = 200$, $N_{\mathrm{test}} = 140$).</p>
                            <p class="text-xs text-slate-500 mt-0.5 lang-vi">COCO val2017 trên RT-DETR-R18VD ($N = 300$, $N_{\mathrm{test}} = 210$) và RT-DETR-R50VD ($N = 200$, $N_{\mathrm{test}} = 140$).</p>
                        </div>
                        <span class="text-[11px] font-mono text-slate-500 bg-white px-2.5 py-1 rounded border border-slate-200 w-fit">
                            $N_{\mathrm{static}} = 30$ &bull; Logit Calib + Enc. Sample
                        </span>
                    </div>

                    <div class="overflow-x-auto custom-scrollbar">
                        <table class="w-full text-left text-xs text-slate-600">
                            <thead class="bg-slate-100/70 border-b border-slate-200 text-[11px] font-bold text-slate-700 uppercase tracking-wider">
                                <tr>
                                    <th class="py-3 px-4"><span class="lang-en">Model &amp; Evaluated Condition</span><span class="lang-vi">Kiến trúc &amp; Điều kiện Đánh giá</span></th>
                                    <th class="py-3 px-3 text-right">COCO mAP@50</th>
                                    <th class="py-3 px-3 text-right">mAP@50:95</th>
                                    <th class="py-3 px-3 text-right">Recall</th>
                                    <th class="py-3 px-3 text-center">Orin Nano Est.</th>
                                    <th class="py-3 px-4"><span class="lang-en">Mechanistic Finding</span><span class="lang-vi">Phát Hiện Cơ Chế</span></th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-100 font-medium text-[11px]">
                                <!-- R18VD -->
                                <tr class="bg-slate-100/60">
                                    <td colspan="6" class="py-1.5 px-4 font-bold text-slate-800 text-[10px] uppercase tracking-wider">
                                        RT-DETR-R18VD (3 Decoder Layers &bull; N = 300, Held-out Adaptive N_test = 210)
                                    </td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900 pl-6">Clean Dynamic-300 (Ref.)</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-slate-900">66.93%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">50.24%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">86.48%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">34.23 ms</td>
                                    <td class="py-2.5 px-4 text-slate-500">Nominal clean reference baseline</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900 pl-6">Clean SQA Logit-Calibrated (330 q.)</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-emerald-700">66.83%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">50.27%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">86.53%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">34.61 ms</td>
                                    <td class="py-2.5 px-4 text-emerald-700">Negligible clean delta (−0.10 pp, +1.08% latency)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-rose-50/20">
                                    <td class="py-2.5 px-4 font-semibold text-rose-950 pl-6">AQSM (ε = 16): Dynamic-300 Baseline</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-rose-600">4.08%</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-rose-600">2.15%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">18.09%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">34.23 ms</td>
                                    <td class="py-2.5 px-4 text-rose-600">Unhardened baseline (6.10% retention)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 text-slate-700 pl-6">Controls: Dyn. Top-330 / Random-30</td>
                                    <td class="py-2.5 px-3 text-right font-mono">4.05% / 4.01%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">2.12% / 2.18%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">18.14% / 18.18%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">34.61 ms</td>
                                    <td class="py-2.5 px-4 text-slate-500">Capacity/random controls yield −0.03 / −0.07 pp</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 text-slate-700 pl-6">SQA Raw Uncalibrated / Logit + Noise</td>
                                    <td class="py-2.5 px-3 text-right font-mono">3.98% / 4.01%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">2.17% / 2.16%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">18.18% / 18.14%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">34.61 ms</td>
                                    <td class="py-2.5 px-4 text-slate-500">Eliminating logit center-bias still gives null recovery</td>
                                </tr>
                                <tr class="hover:bg-sky-50/40 transition bg-sky-50/30">
                                    <td class="py-2.5 px-4 font-bold text-sky-950 pl-6">SQA Logit-Calibrated + Encoder Sample</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-sky-700">4.03%</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-sky-700">2.18%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">17.95%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">34.61 ms</td>
                                    <td class="py-2.5 px-4 text-sky-900 font-semibold">Null (−0.05 pp); proves corrupted $M_{\mathrm{adv}}$ bottleneck</td>
                                </tr>
                                <tr class="hover:bg-amber-50/40 transition">
                                    <td class="py-2.5 px-4 text-slate-700 pl-6">Adaptive Evasion (Calib. Noise / Enc.-Sample, N=210)</td>
                                    <td class="py-2.5 px-3 text-right font-mono">1.79% / 2.65%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">0.77% / 1.63%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">11.39% / 11.32%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">34.61 ms</td>
                                    <td class="py-2.5 px-4 text-amber-800">97.33% / 96.04% adaptive bypass across all $N=210$ test scenes</td>
                                </tr>

                                <!-- R50VD -->
                                <tr class="bg-slate-100/60">
                                    <td colspan="6" class="py-1.5 px-4 font-bold text-slate-800 text-[10px] uppercase tracking-wider">
                                        RT-DETR-R50VD (6 Decoder Layers &bull; N = 200, Held-out Adaptive N_test = 140)
                                    </td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900 pl-6">Clean Dynamic-300 / Clean SQA Hardened</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-emerald-700">74.45% / 74.47%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">57.47% / 57.57%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">92.11% / 92.17%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">71.37 ms</td>
                                    <td class="py-2.5 px-4 text-emerald-700">Zero clean penalty (+0.02 pp, +1.33% latency)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-rose-50/20">
                                    <td class="py-2.5 px-4 font-semibold text-rose-950 pl-6">AQSM (ε = 16): Dynamic-300 Baseline</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-rose-600">10.15%</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-rose-600">5.75%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">27.56%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">70.42 ms</td>
                                    <td class="py-2.5 px-4 text-rose-600">Unhardened baseline (13.63% retention)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 text-slate-700 pl-6">SQA Raw / Logit + Noise (Controls: 10.18% / 10.43%)</td>
                                    <td class="py-2.5 px-3 text-right font-mono">10.10% / 10.29%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">5.56% / 5.82%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">27.63% / 27.98%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">71.37 ms</td>
                                    <td class="py-2.5 px-4 text-slate-500">Logit calibration gives negligible +0.14 pp shift</td>
                                </tr>
                                <tr class="hover:bg-sky-50/40 transition bg-sky-50/30">
                                    <td class="py-2.5 px-4 font-bold text-sky-950 pl-6">SQA Logit-Calibrated + Encoder Sample</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-sky-700">10.27%</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-sky-700">5.81%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">28.12%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">71.37 ms</td>
                                    <td class="py-2.5 px-4 text-sky-900 font-semibold">Null recovery (+0.12 pp); $M_{\mathrm{adv}}$ dominates</td>
                                </tr>
                                <tr class="hover:bg-amber-50/40 transition">
                                    <td class="py-2.5 px-4 text-slate-700 pl-6">Adaptive Evasion (Calib. Noise / Enc.-Sample, N=140)</td>
                                    <td class="py-2.5 px-3 text-right font-mono">8.59% / 6.66%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">5.08% / 3.80%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">18.44% / 18.24%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">71.37 ms</td>
                                    <td class="py-2.5 px-4 text-amber-800">88.46% / 91.06% adaptive bypass across $N=140$ test scenes</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <div class="p-3.5 bg-slate-50/80 border-t border-slate-200 text-[11px] text-slate-500 leading-relaxed">
                        <span class="lang-en"><strong>Mechanistic Lesson:</strong> Zero-shot SQA—even after resolving coordinate logit clustering (<code>torch.logit</code>) and sampling features directly from encoder memory—yields null recovery ($4.08\% \to 4.03\%$ R18VD; $10.15\% \to 10.27\%$ R50VD). Because causal surgery (Arm $\text{C}_{\text{full}}$ retains only $9.31\%$ / $15.46\%$) shows corrupted memory $M_{\mathrm{adv}}$ loses semantic discriminability, encoder representation hardening must accompany calibrated static query priors.</span>
                        <span class="lang-vi"><strong>Bài học Cơ chế:</strong> SQA zero-shot—ngay cả khi xử lý sai lệch tọa độ (<code>torch.logit</code>) và lấy mẫu đặc trưng trực tiếp từ bộ nhớ encoder—vẫn không khôi phục được độ chính xác ($4.08\% \to 4.03\%$ trên R18VD; $10.15\% \to 10.27\%$ trên R50VD). Phù hợp với can thiệp nhân quả (Nhánh $\text{C}_{\text{full}}$ chỉ giữ $9.31\%$ / $15.46\%$), phòng thủ hiệu quả bắt buộc phải kết hợp gia cố biểu diễn của Encoder đa tầng.</span>
                    </div>
                </div>

            </div>
        </section>

        <!-- BibTeX & Citation Section -->
        <section id="bibtex" class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 mb-20">
            <div class="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-sm">
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center gap-2">
                        <span class="w-8 h-8 rounded-lg bg-sky-100 text-sky-700 flex items-center justify-center text-sm">
                            <i class="fa-solid fa-quote-left"></i>
                        </span>
                        <h3 class="text-xl font-bold text-slate-900 font-heading">
                            <span class="lang-en">BibTeX Citation</span>
                            <span class="lang-vi">Trích dẫn BibTeX</span>
                        </h3>
                    </div>
                    <button onclick="copyBibTeX()" class="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 transition">
                        <i class="fa-regular fa-copy"></i>
                        <span id="copyBtnText">
                            <span class="lang-en">Copy BibTeX</span>
                            <span class="lang-vi">Sao chép BibTeX</span>
                        </span>
                    </button>
                </div>
                <div class="relative bg-slate-900 text-slate-100 rounded-2xl p-4 sm:p-5 font-mono text-xs overflow-x-auto custom-scrollbar">
                    <pre id="bibtexBlock"><code>@article{le2026rtdetr_failure_modes,
  title={Decomposing Failure Modes of Real-Time DETR under Encoder Adversarial Attacks: An Empirical Diagnostic Study},
  author={Le, Viet-Anh and Dao, Thuy-Quynh Thi and Nguyen, Trong-Khanh},
  journal={Engineering Applications of Artificial Intelligence},
  year={2026},
  publisher={Elsevier}
}</code></pre>
                </div>
            </div>
        </section>

    </main>

    <!-- Footer -->
    <footer class="bg-white border-t border-slate-200 py-10 text-xs text-slate-500">
        <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
            <div>
                <p class="font-semibold text-slate-700">
                    <span class="lang-en">AI TrustWorthy Research Group</span>
                    <span class="lang-vi">Nhóm Nghiên cứu AI Đáng Tin Cậy (AI TrustWorthy)</span>
                </p>
                <p>
                    <span class="lang-en">Faculty of Information Technology 1 &bull; Posts and Telecommunications Institute of Technology (PTIT), Hanoi, Vietnam</span>
                    <span class="lang-vi">Khoa Công nghệ Thông tin 1 &bull; Học viện Công nghệ Bưu chính Viễn thông (PTIT), Hà Nội, Việt Nam</span>
                </p>
            </div>
            <div class="flex items-center gap-4 text-sm text-slate-400">
                <a href="https://github.com/vietanhlee/AI_TrustWorthy" target="_blank" class="hover:text-slate-900 transition"><i class="fa-brands fa-github"></i></a>
                <a href="mailto:levietanhtrump@gmail.com" class="hover:text-slate-900 transition"><i class="fa-solid fa-envelope"></i></a>
            </div>
        </div>
    </footer>

    <!-- Interactive Metric Popover Element -->
    <div id="metricPopover" class="fixed z-50 bg-slate-900/95 backdrop-blur-md text-white p-3.5 rounded-xl shadow-2xl text-xs max-w-xs transition-opacity duration-200 border border-slate-700/60 hidden pointer-events-auto" style="width: 290px;">
        <div class="flex items-center justify-between border-b border-slate-800 pb-1.5 mb-2">
            <span id="popTitle" class="font-bold text-sky-400 font-heading text-[12px]"></span>
            <button onclick="closePopover()" class="text-slate-400 hover:text-white text-xs px-1" title="Đóng / Close">&times;</button>
        </div>
        <div class="space-y-1.5 text-[11px] leading-relaxed">
            <div>
                <span class="text-slate-400 font-semibold uppercase text-[9px] block">
                    <span class="lang-en">Calculation:</span>
                    <span class="lang-vi">Cách tính:</span>
                </span>
                <span id="popCalc" class="text-slate-200"></span>
            </div>
            <div>
                <span class="text-slate-400 font-semibold uppercase text-[9px] block">
                    <span class="lang-en">Significance:</span>
                    <span class="lang-vi">Ý nghĩa:</span>
                </span>
                <span id="popMean" class="text-slate-300"></span>
            </div>
        </div>
    </div>

    <!-- Image Lightbox Zoom Modal -->
    <div id="imageModal" class="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-md flex items-center justify-center p-4 hidden transition-opacity duration-300" onclick="closeModal()">
        <div class="relative max-w-6xl w-full max-h-[90vh] flex flex-col items-center" onclick="event.stopPropagation()">
            <button onclick="closeModal()" class="absolute -top-10 right-0 text-white/80 hover:text-white text-xl p-2 font-bold focus:outline-none" title="Đóng / Close">
                <i class="fa-solid fa-xmark text-2xl"></i>
            </button>
            <div class="overflow-auto max-h-[82vh] rounded-2xl border border-slate-800 bg-slate-900/60 shadow-2xl flex items-center justify-center p-2">
                <img id="modalImage" src="" alt="" class="max-w-full max-h-[78vh] object-contain rounded-xl select-none">
            </div>
            <p id="modalCaption" class="mt-3 text-xs text-slate-400 text-center font-medium max-w-3xl"></p>
        </div>
    </div>

    <!-- Global Client Script -->
    <script>
    const metricDefs = {
        'memory': {
            title_en: 'Encoder Memory (M)',
            title_vi: 'Encoder Memory (Biểu diễn bộ nhớ M)',
            calc_en: 'Multi-scale spatial feature maps extracted by hybrid encoder: S3 (80x80), S4 (40x40), S5 (20x20), S=8,400 tokens.',
            calc_vi: 'Bản đồ đặc trưng không gian đa tỷ lệ trích xuất từ hybrid encoder: S3 (80x80), S4 (40x40), S5 (20x20), S=8.400 tokens.',
            mean_en: 'Contains semantic contextual representations. Perturbing M accounts for 60.59% (R18VD, N=300) and 68.22% (R50VD, N=200) of detector collapse.',
            mean_vi: 'Chứa biểu diễn ngữ nghĩa của cảnh vật. Làm sai lệch M chiếm tới 60.59% (R18VD, N=300) và 68.22% (R50VD, N=200) nguyên nhân gây sụp đổ mô hình.'
        },
        'query': {
            title_en: 'Query Prior (Q)',
            title_vi: 'Query Prior (Vị trí Tiên nghiệm Q)',
            calc_en: 'Top-300 candidate positions and reference coordinates routed by dynamic query selection head.',
            calc_vi: 'Top-300 vị trí và tọa độ anchor ứng viên được lựa chọn bởi tầng dynamic query selection.',
            mean_en: 'Clean: foreground object centers. Adversarial: anchors misrouted into background noise clusters (39.41% share on R18VD, 31.78% on R50VD).',
            mean_vi: 'Sạch: anchor bao phủ đúng tâm đối tượng. Đối kháng: anchor bị định tuyến sai lệch vào các cụm nền nhiễu (39.41% trên R18VD, 31.78% trên R50VD).'
        },
        'map50': {
            title_en: 'COCO mAP@50 (IoU ≥ 0.50)',
            title_vi: 'COCO mAP@50 (Ngưỡng IoU ≥ 0.50)',
            calc_en: 'Class-wise Average Precision at IoU ≥ 0.50, averaged over all 80 COCO categories on N=300 scenes.',
            calc_vi: 'Tính Average Precision cho từng lớp tại ngưỡng IoU ≥ 0.50, lấy trung bình trên 80 lớp COCO trên N=300 ảnh.',
            mean_en: 'Measures object detection accuracy under standard geometric overlap tolerance (≥50% bounding box intersection).',
            mean_vi: 'Đo lường độ chính xác phát hiện vật thể ở mức dung sai hình học cơ bản (vùng giao bounding box đạt từ 50% trở lên).'
        },
        'map5095': {
            title_en: 'mAP@50:95 (Primary COCO Metric)',
            title_vi: 'mAP@50:95 (Thước đo COCO chuẩn cốt lõi)',
            calc_en: 'Mean Average Precision across 10 IoU thresholds from 0.50 to 0.95 with step 0.05.',
            calc_vi: 'Trung bình cộng của 10 mức AP tại các ngưỡng IoU từ 0.50 đến 0.95 (bước nhảy 0.05).',
            mean_en: 'Comprehensive benchmark metric; simultaneously penalizes classification errors and loose localization.',
            mean_vi: 'Thước đo toàn diện và khắt khe nhất của chuẩn COCO; đánh giá đồng thời độ chính xác phân loại và độ khít của tọa độ hộp.'
        },
        'retention': {
            title_en: 'Relative Retention (%)',
            title_vi: 'Relative Retention (Tỷ lệ duy trì %)',
            calc_en: 'Retention = (mAP_arm / mAP_clean_Arm_A) × 100%.',
            calc_vi: 'Tỷ lệ duy trì = (mAP_nhánh / mAP_gốc_Arm_A) × 100%.',
            mean_en: 'Percentage of nominal baseline capability preserved under counterfactual interventional manipulation.',
            mean_vi: 'Tỷ lệ phần trăm năng lực phát hiện còn bảo toàn sau can thiệp phản thực tế so với mốc sạch ban đầu (Arm A).'
        },
        'attention_entropy': {
            title_en: 'Normalized Attention Entropy H(A)',
            title_vi: 'Entropy Attention Chuẩn hóa H(A)',
            calc_en: 'H(A) = - Σ a_k log(a_k) / log(K*L), computed over L=3 scales and K=4 deformable sampling offsets across all heads.',
            calc_vi: 'H(A) = - Σ a_k log(a_k) / log(K*L), tính toán trên L=3 tầng tỷ lệ và K=4 điểm lấy mẫu biến dạng qua các đầu attention.',
            mean_en: 'Passes ±0.02 TOST equivalence (0.8032 -> 0.7945 R18VD, p=3.46e-5; 0.8150 -> 0.8087 R50VD, p=6.80e-6), refuting attention scattering.',
            mean_vi: 'Đạt kiểm định tương đương TOST ±0.02 (0.8032 -> 0.7945 R18VD, p=3.46e-5; 0.8150 -> 0.8087 R50VD, p=6.80e-6), bác bỏ giả thuyết tán xạ chú ý.'
        },
        'sampling_hit': {
            title_en: 'Spatial Sampling Hit Rate (FG %)',
            title_vi: 'Tỷ lệ Lấy mẫu trúng Foreground (FG %)',
            calc_en: 'Percentage of deformable sampling point locations that fall inside ground-truth foreground object masks.',
            calc_vi: 'Tỷ lệ phần trăm các điểm lấy mẫu biến dạng rơi trúng bên trong vùng đối tượng foreground ground-truth.',
            mean_en: 'Collapses from 63.53% (clean) down to 45.44% under AQSM on R18VD (p = 4.85 × 10^-29), starving cross-attention heads of genuine object evidence.',
            mean_vi: 'Sụp đổ từ 63.53% (ảnh sạch) xuống 45.44% dưới AQSM trên R18VD (p = 4.85 × 10^-29), khiến các đầu cross-attention suy giảm nghiêm trọng tín hiệu đặc trưng đối tượng thật.'
        },
        'ref_drift': {
            title_en: 'Reference Coordinate Drift (Δp)',
            title_vi: 'Độ trôi Tọa độ Tham chiếu (Δp)',
            calc_en: 'Mean displacement ||p_adv - p_clean||_2 of anchor centers, alongside permutation-invariant Hungarian set distance.',
            calc_vi: 'Độ dịch chuyển tâm anchor ||p_adv - p_clean||_2 kết hợp khoảng cách tập hợp bất biến hoán vị Hungarian.',
            mean_en: 'Quantifies spatial diversion into peripheral background noise (Clean: 0.0000, AQSM: 0.4672 R18VD / 0.4459 R50VD).',
            mean_vi: 'Định lượng mức độ trôi lệch không gian vào nhiễu nền ngoại vi (Clean: 0.0000, AQSM: 0.4672 R18VD / 0.4459 R50VD).'
        },
        'dropoff_ratio': {
            title_en: 'Proposal Dropoff Ratio D(X)',
            title_vi: 'Tỷ số Suy giảm Đề xuất D(X)',
            calc_en: 'D(X) = (s_(1) + 10^-4) / (s_(300) + 10^-4), ratio of highest proposal logit to the 300th boundary candidate.',
            calc_vi: 'D(X) = (s_(1) + 10^-4) / (s_(300) + 10^-4), tỷ số giữa logit đề xuất cao nhất và đề xuất thứ 300 tại biên.',
            mean_en: 'Measures confidence distribution sharpness. Clean inference produces steep decay (34.73), while AQSM flattens it (11.21).',
            mean_vi: 'Đo độ dốc phân phối độ tin cậy. Ảnh sạch có phân phối rất dốc (34.73), trong khi AQSM san phẳng phân phối này (11.21).'
        }
    };

    let popoverTimer = null;
    let activePopoverBtn = null;
    let currentPopoverKey = null;

    function showMetricPopover(btn, key) {
        clearTimeout(popoverTimer);
        activePopoverBtn = btn;
        currentPopoverKey = key;
        const def = metricDefs[key];
        if (!def) return;
        const isVi = document.documentElement.lang === 'vi';
        document.getElementById('popTitle').innerText = isVi ? def.title_vi : def.title_en;
        document.getElementById('popCalc').innerText = isVi ? def.calc_vi : def.calc_en;
        document.getElementById('popMean').innerText = isVi ? def.mean_vi : def.mean_en;
        
        const pop = document.getElementById('metricPopover');
        pop.classList.remove('hidden');
        
        const rect = btn.getBoundingClientRect();
        const popWidth = Math.min(320, window.innerWidth - 24);
        let left = rect.left + rect.width / 2 - popWidth / 2;
        if (left < 12) left = 12;
        if (left + popWidth > window.innerWidth - 12) left = window.innerWidth - popWidth - 12;
        
        let top = rect.bottom + 8;
        if (top + 180 > window.innerHeight && rect.top > 190) {
            top = rect.top - 180;
        }
        
        pop.style.width = popWidth + 'px';
        pop.style.left = left + 'px';
        pop.style.top = top + 'px';
    }

    function scheduleClosePopover() {
        popoverTimer = setTimeout(() => {
            closePopover();
        }, 250);
    }

    function closePopover() {
        clearTimeout(popoverTimer);
        const pop = document.getElementById('metricPopover');
        if (pop) pop.classList.add('hidden');
        activePopoverBtn = null;
        currentPopoverKey = null;
    }

    function toggleMetricPopover(btn, key, event) {
        if (event) event.stopPropagation();
        const pop = document.getElementById('metricPopover');
        if (activePopoverBtn === btn && !pop.classList.contains('hidden')) {
            closePopover();
        } else {
            showMetricPopover(btn, key);
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        const pop = document.getElementById('metricPopover');
        if (pop) {
            pop.addEventListener('mouseenter', () => clearTimeout(popoverTimer));
            pop.addEventListener('mouseleave', scheduleClosePopover);
        }
        document.addEventListener('click', (e) => {
            if (!e.target.closest('#metricPopover') && !e.target.closest('.metric-help-btn')) {
                closePopover();
            }
        });
    });

    function copyBibTeX() {
        const text = document.getElementById('bibtexBlock').innerText;
        navigator.clipboard.writeText(text).then(() => {
            const btnText = document.getElementById('copyBtnText');
            btnText.innerHTML = '<span class="lang-en">Copied!</span><span class="lang-vi">Đã sao chép!</span>';
            setTimeout(() => {
                btnText.innerHTML = '<span class="lang-en">Copy BibTeX</span><span class="lang-vi">Sao chép BibTeX</span>';
            }, 2500);
        });
    }

    function openModal(imgEl) {
        if (!imgEl) return;
        const modal = document.getElementById('imageModal');
        const modalImg = document.getElementById('modalImage');
        const caption = document.getElementById('modalCaption');
        const isVi = document.documentElement.lang === 'vi';
        const altVi = imgEl.getAttribute('data-alt-vi');
        const altEn = imgEl.getAttribute('data-alt-en');
        const chosenAlt = isVi ? (altVi || imgEl.alt) : (altEn || imgEl.alt);
        modalImg.src = imgEl.src;
        modalImg.alt = chosenAlt || '';
        caption.textContent = chosenAlt || '';
        modal.classList.remove('hidden');
    }

    function closeModal() {
        const modal = document.getElementById('imageModal');
        modal.classList.add('hidden');
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closePopover();
            closeModal();
        }
    });

    function setLanguage(lang) {
        document.documentElement.lang = lang;
        localStorage.setItem('site_lang', lang);
        const btnEn = document.getElementById('btn-lang-en');
        const btnVi = document.getElementById('btn-lang-vi');
        if (lang === 'vi') {
            btnVi.className = 'px-2.5 py-1 rounded-full transition-all bg-white text-slate-900 shadow-xs';
            btnEn.className = 'px-2.5 py-1 rounded-full transition-all text-slate-500 hover:text-slate-900';
            document.title = 'AI TrustWorthy - Phân Rã Các Kênh Gây Sụp Đổ Của Real-Time DETR Dưới Tấn Công Đối Kháng Vào Encoder';
        } else {
            btnEn.className = 'px-2.5 py-1 rounded-full transition-all bg-white text-slate-900 shadow-xs';
            btnVi.className = 'px-2.5 py-1 rounded-full transition-all text-slate-500 hover:text-slate-900';
            document.title = 'AI TrustWorthy - Decomposing Failure Modes of Real-Time DETR under Encoder Adversarial Attacks';
        }
        if (activePopoverBtn && currentPopoverKey) {
            showMetricPopover(activePopoverBtn, currentPopoverKey);
        }
        if (window.MathJax && MathJax.typesetPromise) {
            MathJax.typesetPromise();
        }
    }

    // Auto-detect saved language on load
    (function() {
        const savedLang = localStorage.getItem('site_lang') || 'en';
        setLanguage(savedLang);
    })();
    </script>
</body>
</html>
'''
    return html_content

def main():
    content = generate_html()
    target_root = "G:/Edge_Sec/index.html"
    target_docs = "G:/Edge_Sec/docs/index.html"
    
    with open(target_root, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {target_root}")
    
    os.makedirs(os.path.dirname(target_docs), exist_ok=True)
    with open(target_docs, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {target_docs}")

if __name__ == "__main__":
    main()
