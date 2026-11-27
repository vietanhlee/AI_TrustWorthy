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
                    <span class="lang-en">R18 vs R50 &amp; Seeds</span><span class="lang-vi">R18 vs R50 &amp; Đa Seed</span>
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
                    <span class="lang-en">Table 13 Benchmark</span>
                    <span class="lang-vi">Đối chiếu Bảng 13</span>
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
                        <span class="lang-en">MS COCO val2017 &bull; N=300 &bull; R18VD &amp; R50VD</span>
                        <span class="lang-vi">MS COCO val2017 &bull; N=300 &bull; R18VD &amp; R50VD</span>
                    </span>
                </div>

                <div class="text-xs sm:text-sm text-slate-600 leading-relaxed mb-4">
                    <p class="lang-en">
                        Real-Time Detection Transformers (RT-DETR) eliminate Non-Maximum Suppression (NMS) via bipartite matching and dynamic top-$K$ query selection atop a multi-scale hybrid encoder. Because proposal selection operates directly on encoder feature pyramids, white-box encoder perturbations simultaneously corrupt memory representations and query routing. We present a symmetric empirical diagnostic study across both <strong>RT-DETR-R18VD</strong> (3 decoder layers) and <strong>RT-DETR-R50VD</strong> (6 decoder layers) on COCO val2017 ($N = 300$, native $640 \times 640$, $S = 8,400$ tokens, $\ell_\infty$ budget $\epsilon = 16/255$), using Adversarial Query Selection Manipulation (AQSM) as an encoder-restricted probe within a $2 \times 2$ factorial tensor-swap protocol alongside Query Vanishing and Standard Task PGD baselines.
                    </p>
                    <p class="lang-vi">
                        Mô hình Real-Time Detection Transformer (RT-DETR) loại bỏ thuật toán Non-Maximum Suppression (NMS) bằng phép so khớp hai phía (bipartite matching) và cơ chế chọn lọc truy vấn động top-$K$ trên kim tự tháp đặc trưng đa tầng. Do tầng lựa chọn đề xuất hoạt động trực tiếp trên đặc trưng Encoder, các nhiễu loạn đối kháng white-box đồng thời làm sai lệch cả biểu diễn bộ nhớ lẫn định tuyến vị trí truy vấn. Chúng tôi thực hiện nghiên cứu chẩn đoán thực nghiệm đối xứng trên cả <strong>RT-DETR-R18VD</strong> (3 tầng decoder) và <strong>RT-DETR-R50VD</strong> (6 tầng decoder) trên tập COCO val2017 ($N = 300$, độ phân giải gốc $640 \times 640$, $S = 8.400$ tokens, ngưỡng nhiễu $\epsilon = 16/255$), triển khai AQSM như một công cụ chẩn đoán giới hạn tại Encoder trong giao thức can thiệp hoán đổi tensor $2 \times 2$ cùng các mốc chuẩn Query Vanishing và Task PGD.
                    </p>
                </div>

                <!-- 4 Core Findings Grid -->
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs">
                    <div class="p-3 rounded-xl bg-slate-50/80 border border-slate-200/60 flex items-start gap-2.5">
                        <span class="w-5 h-5 rounded-md bg-rose-100 text-rose-600 flex items-center justify-center text-[10px] flex-shrink-0 mt-0.5 font-bold">1</span>
                        <div>
                            <strong class="text-slate-900 block font-semibold mb-0.5">
                                <span class="lang-en">Protocol-Level Failure Decomposition:</span>
                                <span class="lang-vi">Phân Rã Kênh Gây Sụp Đổ:</span>
                            </strong>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-en">AQSM collapses mAP@50 from $66.93\%$ to $4.05\%$ on R18VD (5-seed $4.00 \pm 0.74\%$) and $72.95\% \to 9.14\%$ on R50VD (5-seed $10.19 \pm 1.10\%$). Memory corruption receives $60.59\%$ per-image Shapley share on R18VD ($66.75\%$ pooled mAP, $71.40\%$ log-scale), shifting to $68.16\%$ on 6-layer R50VD; query eviction is an essential secondary channel ($39.41\%$ / $31.84\%$).</span>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-vi">AQSM làm sụp đổ mAP@50 từ $66.93\%$ xuống $4.05\%$ trên R18VD (5-seed $4.00 \pm 0.74\%$) và từ $72.95\%$ xuống $9.14\%$ trên R50VD (5-seed $10.19 \pm 1.10\%$). Tổn hại bộ nhớ nhận $60.59\%$ tỷ trọng Shapley trên R18VD ($66.75\%$ pooled mAP, $71.40\%$ log-scale) và tăng lên $68.16\%$ trên R50VD 6 tầng; trục xuất query là kênh thứ cấp thiết yếu ($39.41\%$ / $31.84\%$).</span>
                        </div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-50/80 border border-slate-200/60 flex items-start gap-2.5">
                        <span class="w-5 h-5 rounded-md bg-amber-100 text-amber-600 flex items-center justify-center text-[10px] flex-shrink-0 mt-0.5 font-bold">2</span>
                        <div>
                            <strong class="text-slate-900 block font-semibold mb-0.5">
                                <span class="lang-en">Attention Localization vs Spatial Misrouting:</span>
                                <span class="lang-vi">Tiêu Điểm Chú Ý vs Định Tuyến Sai Lệch:</span>
                            </strong>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-en">Normalized deformable attention entropy remains stable $\approx 0.80$ ($0.803 \to 0.794$ R18VD; $0.816 \to 0.807$ R50VD), refuting attention scattering. Instead, reference coordinates drift into background clutter ($\Delta p = 0.475$ R18VD; $0.450$ R50VD), driving anchor FG hit rate down ($66.0\% \to 45.4\%$, TOST equivalent to uniform sampling $p = 0.048$).</span>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-vi">Entropy attention chuẩn hóa duy trì mức hẹp $\approx 0.80$ ($0.803 \to 0.794$ R18VD; $0.816 \to 0.807$ R50VD), bác bỏ giả thuyết tán xạ chú ý. Ngược lại, tọa độ tham chiếu trôi dạt vào vùng nền nhiễu ($\Delta p = 0.475$ R18VD; $0.450$ R50VD), kéo tỷ lệ trúng FG ($66.0\% \to 45.4\%$, kiểm định TOST tương đương lấy mẫu đều $p = 0.048$).</span>
                        </div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-50/80 border border-slate-200/60 flex items-start gap-2.5">
                        <span class="w-5 h-5 rounded-md bg-sky-100 text-sky-600 flex items-center justify-center text-[10px] flex-shrink-0 mt-0.5 font-bold">3</span>
                        <div>
                            <strong class="text-slate-900 block font-semibold mb-0.5">
                                <span class="lang-en">Fixed-Query Hardware Latency Stability:</span>
                                <span class="lang-vi">Độ Trễ Phần Cứng Đơn Định Cố Định Query:</span>
                            </strong>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-en">Timed via CUDA events on NVIDIA Tesla T4 (FP32), RT-DETR exhibits invariant latency across Clean, AQSM, Task PGD, and a 207-box proposal flooding sponge stress test ($39.02 \to 38.66$ ms R18VD; $74.89 \to 72.76$ ms R50VD), confirming structural immunity to NMS latency spikes.</span>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-vi">Đo đạc bằng CUDA events trên NVIDIA Tesla T4 (FP32), RT-DETR giữ vững độ trễ suy luận bất biến qua các điều kiện Clean, AQSM, Task PGD và thử nghiệm bẫy trễ proposal sponge 207 hộp ($39.02 \to 38.66$ ms R18VD; $74.89 \to 72.76$ ms R50VD), khẳng định miễn nhiễm cấu trúc trước bẫy trễ NMS.</span>
                        </div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-50/80 border border-slate-200/60 flex items-start gap-2.5">
                        <span class="w-5 h-5 rounded-md bg-indigo-100 text-indigo-600 flex items-center justify-center text-[10px] flex-shrink-0 mt-0.5 font-bold">4</span>
                        <div>
                            <strong class="text-slate-900 block font-semibold mb-0.5">
                                <span class="lang-en">Defense Limits &amp; Negative SQA Result:</span>
                                <span class="lang-vi">Ranh Giới Phòng Thủ &amp; Bài Học Từ SQA:</span>
                            </strong>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-en">Runtime monitor ESM (AUROC $0.7881$ R18VD / $0.6914$ R50VD) suffers $94.76\%\text{--}97.14\%$ raw bypass under white-box adaptive evasion. Zero-shot Static Query Augmentation (SQA) yields an informative empirical null result ($4.05\% \to 3.97\%$ R18VD; $9.14\% \to 9.25\%$ R50VD), proving unhardened static queries cannot rescue severely corrupted encoder memory ($M_{\mathrm{adv}}$).</span>
                            <span class="text-slate-600 text-[11px] leading-relaxed lang-vi">Bộ giám sát ESM (AUROC $0.7881$ R18VD / $0.6914$ R50VD) bị đối thủ thích nghi qua mặt $94.76\%\text{--}97.14\%$. Kỹ thuật Tăng cường Truy vấn Tĩnh (SQA) zero-shot đem lại kết quả null mang tính chẩn đoán ($4.05\% \to 3.97\%$ R18VD; $9.14\% \to 9.25\%$ R50VD), chứng minh query tĩnh chưa gia cố không thể khôi phục bộ nhớ encoder đã bị tàn phá nặng ($M_{\mathrm{adv}}$).</span>
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
                            Real-Time Detection Transformers (RT-DETR) eliminate Non-Maximum Suppression (NMS) via bipartite matching and dynamic top-$K$ query selection atop a multi-scale hybrid encoder. Because proposal selection operates directly on encoder feature pyramids, white-box encoder perturbations simultaneously corrupt memory representations and query routing. We present a symmetric empirical diagnostic study across both RT-DETR-R18VD (3 decoder layers) and RT-DETR-R50VD (6 decoder layers) on COCO val2017 ($N = 300$, native $640 \times 640$, $S = 8,400$ tokens, $\ell_\infty$ budget $\epsilon = 16/255$), using Adversarial Query Selection Manipulation (AQSM) as an encoder-restricted probe within a $2 \times 2$ factorial tensor-swap protocol alongside Query Vanishing and Standard Task PGD baselines.
                        </p>
                        <p class="lang-en">
                            Our measurements establish four main observations: (1) Protocol-Level Failure Decomposition: AQSM collapses COCO mAP@50 from $66.93\%$ to $4.05\%$ on R18VD ($-93.96\%$; 5-seed fresh generation $4.00 \pm 0.74\%$) and from $72.95\%$ to $9.14\%$ on R50VD ($-87.47\%$; 5-seed $10.19 \pm 1.10\%$). Under our $2 \times 2$ tensor-swap protocol ($N_{\mathrm{valid}} = 299$), encoder memory corruption receives $60.59\%$ ($95\%$ BCa CI $[58.33\%, 62.84\%]$) per-image Shapley attribution on $AP_i$ ($66.75\%$ pooled COCO mAP; $71.40\%$ log-retention scale) vs. $39.41\%$ for query prior disruption on R18VD, shifting further toward memory on 6-layer R50VD ($68.16\%$ per-image / $72.19\%$ pooled / $75.20\%$ log-retention). (2) Attention Localization vs. Spatial Misrouting: Across decoder layers, normalized deformable attention entropy stays within a narrow band near $\approx 0.80$ ($0.8032 \to 0.7938$ on R18VD; $0.8156 \to 0.8071$ on R50VD), whereas large reference coordinate drift drops anchor foreground hit rate ($66.00\% \to 45.37\%$ R18VD; $67.02\% \to 49.67\%$ R50VD) and deformable sampling hit rate ($63.53\% \to 44.17\%$ R18VD), approaching uniform spatial chance ($45.02\%$ uniform grid / $44.75\%$ GT union) across AQSM, Vanishing, and Task PGD. (3) Measured Execution Latency Stability: Timed via CUDA events on NVIDIA Tesla T4 (PyTorch FP32), RT-DETR exhibits invariant latency across Clean, AQSM, Task PGD, and a 207-box proposal-flooding sponge attack ($39.02 \to 38.66$ ms on R18VD; $74.89 \to 72.76$ ms on R50VD), consistent with fixed-query $N_q = 300$ NMS-free execution. (4) Defense Baseline Limits and Negative SQA Result: A calibrated Encoder Selection Monitor (ESM, stationary AUROC $0.7881$ R18VD / $0.6914$ R50VD) suffers $94.76\%\text{--}97.14\%$ raw monitor bypass under white-box adaptive evasion, while zero-shot Static Query Augmentation (SQA) fails to restore stationary detection ($4.05\% \to 3.97\%$, $-0.08$ pp R18VD; $9.14\% \to 9.25\%$, $+0.11$ pp R50VD), demonstrating that unhardened query augmentation cannot compensate for corrupted encoder memory ($M_{\mathrm{adv}}$).
                        </p>
                        <p class="lang-vi">
                            Mô hình Real-Time Detection Transformer (RT-DETR) loại bỏ bước hậu xử lý phi cực đại (NMS) bằng cơ chế bipartite matching và chọn lọc truy vấn động top-$K$ trên đỉnh bộ mã hóa lai đa tầng. Vì tầng lựa chọn đề xuất hoạt động trực tiếp trên kim tự tháp đặc trưng, các nhiễu loạn đối kháng white-box đồng thời làm sai lệch cả biểu diễn bộ nhớ lẫn định tuyến vị trí truy vấn. Chúng tôi trình bày một nghiên cứu chẩn đoán thực nghiệm đối xứng trên cả RT-DETR-R18VD (3 tầng decoder) và RT-DETR-R50VD (6 tầng decoder) trên tập COCO val2017 ($N = 300$, $640 \times 640$, $S = 8.400$ tokens, $\epsilon = 16/255$), sử dụng AQSM làm công cụ chẩn đoán giới hạn tại Encoder trong giao thức can thiệp hoán đổi tensor $2 \times 2$ cùng các mốc chuẩn Query Vanishing và Task PGD.
                        </p>
                        <p class="lang-vi">
                            Kết quả thực nghiệm xác lập 4 quan sát cốt lõi: (1) Phân rã thất bại cấp giao thức: AQSM kéo tụt COCO mAP@50 từ $66.93\%$ xuống $4.05\%$ trên R18VD ($-93.96\%$; sinh mới 5 seed $4.00 \pm 0.74\%$) và từ $72.95\%$ xuống $9.14\%$ trên R50VD ($-87.47\%$; 5-seed $10.19 \pm 1.10\%$). Dưới giao thức hoán đổi tensor $2 \times 2$ ($N_{\mathrm{valid}} = 299$), tổn hại bộ nhớ encoder chiếm $60.59\%$ (95% BCa CI $[58.33\%, 62.84\%]$) đóng góp Shapley từng ảnh trên $AP_i$ ($66.75\%$ pooled COCO mAP; $71.40\%$ trên thang log-retention) so với $39.41\%$ cho sự phá vỡ vị trí query trên R18VD, và dịch chuyển mạnh hơn về phía bộ nhớ trên R50VD 6 tầng ($68.16\%$ per-image / $72.19\%$ pooled / $75.20\%$ log-retention). (2) Tiêu điểm chú ý vs Định tuyến sai lệch: Qua các tầng decoder, entropy attention chuẩn hóa biến dạng nằm ổn định trong dải hẹp quanh $\approx 0.80$ ($0.8032 \to 0.7938$ trên R18VD; $0.8156 \to 0.8071$ trên R50VD), trong khi độ trôi tọa độ tham chiếu kéo tụt tỷ lệ anchor trúng foreground ($66.00\% \to 45.37\%$ R18VD; $67.02\% \to 49.67\%$ R50VD) và tỷ lệ lấy mẫu trúng ($63.53\% \to 44.17\%$ R18VD), tiệm cận xác suất lấy mẫu đều ngẫu nhiên ($45.02\%$ lưới đều / $44.75\%$ hợp GT) trên cả AQSM, Vanishing và Task PGD. (3) Tính ổn định độ trễ thực thi phần cứng: Đo đạc bằng CUDA events trên NVIDIA Tesla T4 (PyTorch FP32), RT-DETR duy trì độ trễ bất biến giữa Clean, AQSM, Task PGD và thử nghiệm bẫy trễ proposal sponge 207 hộp ($39.02 \to 38.66$ ms trên R18VD; $74.89 \to 72.76$ ms trên R50VD), phù hợp với kiến trúc cố định $N_q = 300$ không dùng NMS. (4) Ranh giới phòng thủ và kết quả null của SQA: Bộ giám sát ESM (AUROC tĩnh $0.7881$ R18VD / $0.6914$ R50VD) bị qua mặt $94.76\%\text{--}97.14\%$ dưới đối thủ thích nghi white-box, trong khi kỹ thuật Tăng cường Truy vấn Tĩnh (SQA) zero-shot không thể khôi phục độ chính xác ($4.05\% \to 3.97\%$, $-0.08$ pp R18VD; $9.14\% \to 9.25\%$, $+0.11$ pp R50VD), chứng minh query tĩnh chưa gia cố không thể bù đắp cho bộ nhớ encoder đã bị sai lệch ($M_{\mathrm{adv}}$).
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
                    Surgically swapping memory ($M$) and query ($Q$) representations across clean and adversarial passes isolates and quantifies their contributions to detector collapse across both RT-DETR-R18VD and RT-DETR-R50VD ($N = 300$, $N_{\mathrm{valid}} = 299$, $\epsilon = 16/255$).
                </p>
                <p class="text-sm text-slate-600 lang-vi">
                    Can thiệp hoán đổi có kiểm soát giữa biểu diễn bộ nhớ ($M$) và truy vấn ($Q$) qua các lượt chạy sạch và đối kháng giúp cô lập và định lượng chính xác từng kênh gây sụp đổ mô hình trên cả RT-DETR-R18VD và RT-DETR-R50VD ($N = 300$, $N_{\mathrm{valid}} = 299$, $\epsilon = 16/255$).
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
                        COCO val2017 &bull; N=300 &bull; Native 640×640 &bull; ε=16/255
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
                                    RT-DETR-R50VD (6 Decoder Layers &bull; ResNet-50 Backbone &bull; 42M Params)
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
                                <td class="py-3 px-4 text-right font-bold text-slate-900 font-mono">72.95%</td>
                                <td class="py-3 px-4 text-right font-mono">55.73%</td>
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
                                <td class="py-3 px-4 text-right font-bold text-rose-600 font-mono">9.14%</td>
                                <td class="py-3 px-4 text-right font-mono text-rose-600">5.41%</td>
                                <td class="py-3 px-5 text-right text-rose-600 font-bold font-mono">12.5%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition">
                                <td class="py-3 px-5 font-semibold text-slate-900 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
                                    <span class="lang-en">Arm $\text{C}_{\text{full}}$: Adv Memory + Clean Prior</span>
                                    <span class="lang-vi">Nhánh $\text{C}_{\text{full}}$: Bộ nhớ Đối kháng + Query Sạch</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-rose-700 font-semibold">Adversarial</td>
                                <td class="py-3 px-3 text-center font-mono text-emerald-700 font-semibold">Clean (300)</td>
                                <td class="py-3 px-4 text-right font-bold text-slate-900 font-mono">14.43%</td>
                                <td class="py-3 px-4 text-right font-mono">9.02%</td>
                                <td class="py-3 px-5 text-right text-slate-600 font-bold font-mono">19.8%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-slate-50/40">
                                <td class="py-3 px-5 font-semibold text-slate-700 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-slate-400"></span>
                                    <span class="lang-en">Arm $\text{C}_{\text{index}}$: Adv Memory + Clean Index</span>
                                    <span class="lang-vi">Nhánh $\text{C}_{\text{index}}$: Bộ nhớ ĐK + Chỉ số Sạch</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-rose-700 font-semibold">Adversarial</td>
                                <td class="py-3 px-3 text-center font-mono text-purple-700 font-semibold">Adv Feat @ Clean Idx</td>
                                <td class="py-3 px-4 text-right font-bold text-slate-700 font-mono">11.06%</td>
                                <td class="py-3 px-4 text-right font-mono">6.46%</td>
                                <td class="py-3 px-5 text-right text-slate-500 font-bold font-mono">15.2%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition">
                                <td class="py-3 px-5 font-semibold text-slate-900 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-sky-500"></span>
                                    <span class="lang-en">Arm $\text{D}_{\text{full}}$: Clean Memory + Adv Prior</span>
                                    <span class="lang-vi">Nhánh $\text{D}_{\text{full}}$: Bộ nhớ Sạch + Query Đối kháng</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-emerald-700 font-semibold">Clean</td>
                                <td class="py-3 px-3 text-center font-mono text-rose-700 font-semibold">Adversarial (300)</td>
                                <td class="py-3 px-4 text-right font-bold text-slate-900 font-mono">42.75%</td>
                                <td class="py-3 px-4 text-right font-mono">32.25%</td>
                                <td class="py-3 px-5 text-right text-sky-600 font-bold font-mono">58.6%</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-indigo-50/30">
                                <td class="py-3 px-5 font-semibold text-indigo-950 flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-indigo-500"></span>
                                    <span class="lang-en">Arm $\text{D}_{\text{index}}$: Clean Memory + Adv Index</span>
                                    <span class="lang-vi">Nhánh $\text{D}_{\text{index}}$: Bộ nhớ Sạch + Chỉ số Đối kháng</span>
                                </td>
                                <td class="py-3 px-3 text-center font-mono text-emerald-700 font-semibold">Clean</td>
                                <td class="py-3 px-3 text-center font-mono text-purple-700 font-semibold">Clean Feat @ Adv Idx</td>
                                <td class="py-3 px-4 text-right font-bold text-indigo-700 font-mono">53.57%</td>
                                <td class="py-3 px-4 text-right font-mono text-indigo-700">41.60%</td>
                                <td class="py-3 px-5 text-right text-indigo-700 font-bold font-mono">73.4%</td>
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
                        <div class="text-2xl font-extrabold text-slate-900 font-mono mt-1">60.59% &rarr; 68.16%</div>
                        <p class="text-xs text-slate-500 mt-1.5 leading-relaxed lang-en">
                            Per-image Shapley share on $AP_i$ is <strong>60.59%</strong> on R18VD (CI $[58.33, 62.84]$%; pooled $66.75\%$; log-scale $71.40\%$), shifting to <strong>68.16%</strong> on 6-layer R50VD (pooled $72.19\%$). Repeated-measures ANOVA shows $\text{ME}_M = -42.95$ pp ($\eta_p^2 = 0.784, F = 1084.1, p < 0.001$).
                        </p>
                        <p class="text-xs text-slate-500 mt-1.5 leading-relaxed lang-vi">
                            Tỷ trọng Shapley từng ảnh trên $AP_i$ đạt <strong>60.59%</strong> trên R18VD (CI $[58.33, 62.84]$%; pooled $66.75\%$; log-scale $71.40\%$), tăng lên <strong>68.16%</strong> trên R50VD (pooled $72.19\%$). ANOVA đo lường lặp lại xác nhận $\text{ME}_M = -42.95$ pp ($\eta_p^2 = 0.784, F = 1084.1, p < 0.001$).
                        </p>
                    </div>
                </div>

                <div class="p-4 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col justify-between">
                    <div>
                        <span class="text-[10px] font-bold uppercase tracking-wider text-sky-600">
                            <span class="lang-en">Secondary Channel: Query ($Q$)</span>
                            <span class="lang-vi">Kênh Thứ Cấp: Query Prior ($Q$)</span>
                        </span>
                        <div class="text-2xl font-extrabold text-slate-900 font-mono mt-1">39.41% &rarr; 31.84%</div>
                        <p class="text-xs text-slate-500 mt-1.5 leading-relaxed lang-en">
                            Query eviction accounts for <strong>39.41%</strong> Shapley share on R18VD (pooled $33.25\%$, log-scale $28.60\%$; $\text{ME}_Q = -27.94$ pp, $\eta_p^2 = 0.709, F = 725.2$), buffering down to <strong>31.84%</strong> on 6-layer R50VD ($\text{ME}_Q = -19.96$ pp, $\eta_p^2 = 0.637$).
                        </p>
                        <p class="text-xs text-slate-500 mt-1.5 leading-relaxed lang-vi">
                            Trục xuất query chiếm <strong>39.41%</strong> tỷ trọng Shapley trên R18VD (pooled $33.25\%$, log-scale $28.60\%$; $\text{ME}_Q = -27.94$ pp, $\eta_p^2 = 0.709, F = 725.2$), giảm xuống <strong>31.84%</strong> trên R50VD 6 tầng ($\text{ME}_Q = -19.96$ pp, $\eta_p^2 = 0.637$).
                        </p>
                    </div>
                </div>

                <div class="p-4 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col justify-between">
                    <div>
                        <span class="text-[10px] font-bold uppercase tracking-wider text-indigo-600">
                            <span class="lang-en">Clean-Memory Index Resilience ($\text{Arm D}_{\text{index}}$)</span>
                            <span class="lang-vi">Sức Chống Chịu Chỉ Số Khi Bộ Nhớ Sạch</span>
                        </span>
                        <div class="text-2xl font-extrabold text-indigo-700 font-mono mt-1">39.11% &rarr; 53.57%</div>
                        <p class="text-xs text-slate-500 mt-1.5 leading-relaxed lang-en">
                            With pristine memory $M_{\mathrm{clean}}$, purely swapping top-300 indices retains <strong>58.4%</strong> ($39.11\%$ mAP@50) on R18VD, rising to <strong>73.4%</strong> ($53.57\%$ mAP@50) on 6-layer R50VD, showing deeper decoders buffer spatial misindexing.
                        </p>
                        <p class="text-xs text-slate-500 mt-1.5 leading-relaxed lang-vi">
                            Khi giữ nguyên bộ nhớ sạch $M_{\mathrm{clean}}$, chỉ riêng tráo chỉ số top-300 vẫn giữ được <strong>58.4%</strong> ($39.11\%$ mAP@50) trên R18VD và tăng lên <strong>73.4%</strong> ($53.57\%$ mAP@50) trên R50VD, cho thấy decoder sâu hơn có khả năng bù đắp chỉ số tốt hơn.
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
                        <p class="text-[11px] text-slate-500 mb-3 lang-en">Boundary logit $s_{(300)}$ inflates $2.90\times$ ($0.0348 \to 0.1009$), compressing foreground separation.</p>
                        <p class="text-[11px] text-slate-500 mb-3 lang-vi">Điểm tại biên $s_{(300)}$ phình to $2.90\times$ ($0.0348 \to 0.1009$), nén chặt khoảng cách giữa foreground và nền.</p>
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
                        <p class="text-xs text-slate-500 mb-3 lang-en">Multi-scale cosine degradation across $S_3, S_4, S_5$ ($0.6140$ overall R18VD / $0.6758$ R50VD) and query eviction.</p>
                        <p class="text-xs text-slate-500 mb-3 lang-vi">Suy giảm cosine đa tầng qua $S_3, S_4, S_5$ ($0.6140$ R18VD / $0.6758$ R50VD) và sụp đổ độ trùng lặp query.</p>
                    </div>
                    <div class="relative group rounded-2xl overflow-hidden border border-slate-200 bg-slate-50 cursor-zoom-in my-auto" onclick="openModal(this.querySelector('img'))">
                        <img src="assets/images/fig5_multiscale_cosine_and_overlap.png" alt="Figure 5: Multi-scale feature cosine similarity and query overlap dynamics" data-alt-en="Figure 5: Multi-scale feature cosine similarity and query overlap dynamics on COCO val2017" data-alt-vi="Hình 5: Độ tương đồng Cosine đặc trưng đa tầng và động thái suy giảm độ trùng lặp query trên COCO val2017" class="w-full h-auto object-contain transition-transform duration-300 group-hover:scale-[1.015]">
                        <div class="absolute bottom-2.5 right-2.5 px-2 py-0.5 rounded-md bg-slate-900/70 text-white text-[10px] font-medium backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1 pointer-events-none">
                            <i class="fa-solid fa-magnifying-glass-plus"></i>
                            <span class="lang-en">Expand</span><span class="lang-vi">Phóng to</span>
                        </div>
                    </div>
                    <div class="mt-3 pt-3 border-t border-slate-100 text-[11px] text-slate-500 leading-relaxed">
                        <span class="lang-en"><strong>Key Telemetry:</strong> Top-300 proposal overlap collapses to $16.11\%$ on R18VD ($21.40\%$ on R50VD) while $97.22\%$ ($91.76\%$) of top-15 foreground candidate slots are evicted into peripheral background noise.</span>
                        <span class="lang-vi"><strong>Bằng chứng cốt lõi:</strong> Độ trùng lặp top-300 query sụp đổ còn $16.11\%$ trên R18VD ($21.40\%$ trên R50VD) trong khi $97.22\%$ ($91.76\%$) khe ứng viên foreground top-15 bị trục xuất vào nhiễu nền ngoại vi.</span>
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
                        <span class="lang-en"><strong>Key Telemetry:</strong> Normalized entropy remains stable ($0.8032 \to 0.7938$, $d = -0.19$ on R18VD; $0.8156 \to 0.8071$ on R50VD). Conversely, deformable sampling hit drops from $63.53\%$ to $44.17\%$ on R18VD ($p = 2.54 \times 10^{-32}$) and $63.32\% \to 48.31\%$ on R50VD.</span>
                        <span class="lang-vi"><strong>Bằng chứng cốt lõi:</strong> Entropy attention chuẩn hóa gần như bất biến ($0.8032 \to 0.7938$, $d = -0.19$ trên R18VD; $0.8156 \to 0.8071$ trên R50VD). Ngược lại, tỷ lệ lấy mẫu trúng foreground sụp đổ từ $63.53\%$ xuống $44.17\%$ trên R18VD ($p = 2.54 \times 10^{-32}$) và $63.32\% \to 48.31\%$ trên R50VD.</span>
                    </div>
                </div>
            </div>

            <!-- 3 Diagnostic Takeaway Cards -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                <div class="p-4 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col justify-between">
                    <div>
                        <h4 class="font-bold text-slate-900 text-sm mb-1 font-heading flex items-center gap-2">
                            <i class="fa-solid fa-check text-emerald-500"></i>
                            <span class="lang-en">Attention Weights Do Not Disperse</span>
                            <span class="lang-vi">Trọng Số Attention Không Hề Tán Xạ</span>
                        </h4>
                        <p class="text-xs text-slate-500 leading-relaxed lang-en">
                            Normalized deformable attention entropy changes minimally ($0.8032 \to 0.7938$, $\Delta = -0.0094, d = -0.19$ on R18VD; $0.8156 \to 0.8071$ on R50VD). Softmax weights stay sharply localized rather than scattering into diffuse noise ($H \to 1.0$).
                        </p>
                        <p class="text-xs text-slate-500 leading-relaxed lang-vi">
                            Entropy attention biến dạng chuẩn hóa thay đổi tối thiểu ($0.8032 \to 0.7938$, $\Delta = -0.0094, d = -0.19$ trên R18VD; $0.8156 \to 0.8071$ trên R50VD). Trọng số softmax vẫn tập trung cục bộ thay vì phân tán thành nhiễu mờ ($H \to 1.0$).
                        </p>
                    </div>
                    <div class="mt-3 pt-2 border-t border-slate-100 text-[11px] font-mono text-emerald-600 font-semibold">
                        ΔH(A) = -0.0094 (d = -0.19, R18VD)
                    </div>
                </div>

                <div class="p-4 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col justify-between">
                    <div>
                        <h4 class="font-bold text-slate-900 text-sm mb-1 font-heading flex items-center gap-2">
                            <i class="fa-solid fa-crosshairs text-rose-500"></i>
                            <span class="lang-en">Spatial Target Misrouting</span>
                            <span class="lang-vi">Định Tuyến Sai Lệch Tọa Độ Không Gian</span>
                        </h4>
                        <p class="text-xs text-slate-500 leading-relaxed lang-en">
                            Initial reference coordinates undergo large displacement ($\Delta p = 0.4750 \pm 0.0669$ R18VD; $0.4503 \pm 0.0680$ R50VD), starving decoder cross-attention heads of foreground features and pulling sampling hit rate down to $44.17\%$ ($-19.4$ pp).
                        </p>
                        <p class="text-xs text-slate-500 leading-relaxed lang-vi">
                            Tọa độ tham chiếu ban đầu chịu sự dịch chuyển lớn ($\Delta p = 0.4750 \pm 0.0669$ R18VD; $0.4503 \pm 0.0680$ R50VD), khiến các đầu cross-attention thiếu hụt đặc trưng foreground và kéo tỷ lệ lấy mẫu trúng xuống $44.17\%$ (giảm $-19.4$ pp).
                        </p>
                    </div>
                    <div class="mt-3 pt-2 border-t border-slate-100 text-[11px] font-mono text-rose-600 font-semibold">
                        <span class="lang-en">Δp = 0.4750 &bull; Hit: 44.17% (p &lt; 10⁻³¹)</span>
                        <span class="lang-vi">Δp = 0.4750 &bull; Trúng FG: 44.17% (p &lt; 10⁻³¹)</span>
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
                                <td class="py-2.5 px-3 text-center font-mono">0.8320 ± 0.0305</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-600">50.30%</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-600">0.4295 ± 0.0517</td>
                                <td class="py-2.5 px-3 text-center font-mono">17.27 ± 11.74</td>
                                <td class="py-2.5 px-4 text-right font-mono text-rose-600 font-bold">1.32% (2.0%)</td>
                                <td class="py-2.5 px-4 text-amber-800">Proposal suppression; complete scene blackout ($0.67$ boxes)</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition">
                                <td class="py-2.5 px-4 font-semibold text-orange-950 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-orange-500"></span>
                                    Standard Task PGD ($T=15$)
                                </td>
                                <td class="py-2.5 px-3 text-center font-mono">0.7832 ± 0.0342</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-600">47.29%</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-600">0.4536 ± 0.0578</td>
                                <td class="py-2.5 px-3 text-center font-mono">14.65 ± 8.64</td>
                                <td class="py-2.5 px-4 text-right font-mono text-rose-600 font-bold">0.35% (0.5%)</td>
                                <td class="py-2.5 px-4 text-orange-800">Dense false-alarm proposal flooding ($46.55$ boxes vs clean $8.54$)</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-rose-50/40">
                                <td class="py-2.5 px-4 font-bold text-rose-950 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-rose-600"></span>
                                    AQSM Probe ($T=15$)
                                </td>
                                <td class="py-2.5 px-3 text-center font-mono font-bold text-emerald-700">0.7938 ± 0.0293</td>
                                <td class="py-2.5 px-3 text-center font-mono font-bold text-rose-700">44.17%</td>
                                <td class="py-2.5 px-3 text-center font-mono font-bold text-rose-700">0.4750 ± 0.0669</td>
                                <td class="py-2.5 px-3 text-center font-mono font-bold text-rose-700">10.54 ± 7.10</td>
                                <td class="py-2.5 px-4 text-right font-mono font-bold text-rose-600">4.05% (6.0%)</td>
                                <td class="py-2.5 px-4 text-rose-900 font-semibold">Attention entropy invariant ($d=-0.19$); spatial coordinate drift into noise</td>
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
                                <td class="py-2.5 px-4 text-right font-mono font-bold text-sky-800">63.21 ± 0.47% (94.4%)</td>
                                <td class="py-2.5 px-4 text-sky-900">Permuting anchors across queries has minor impact (target boxes still covered)</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-sky-50/30">
                                <td class="py-2.5 px-4 font-semibold text-sky-950 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-sky-500"></span>
                                    Random Shift ($L_2$, 5 seeds)
                                </td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">—</td>
                                <td class="py-2.5 px-3 text-center font-mono text-sky-800">46.19% (Anchor FG)</td>
                                <td class="py-2.5 px-3 text-center font-mono text-sky-800">0.2003 (Hung.)</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">—</td>
                                <td class="py-2.5 px-4 text-right font-mono font-bold text-sky-700">32.06 ± 0.82% (47.9%)</td>
                                <td class="py-2.5 px-4 text-sky-900">Matched isotropic shift halves accuracy, confirming coordinate sensitivity</td>
                            </tr>
                            <tr class="hover:bg-slate-50/60 transition bg-sky-50/50">
                                <td class="py-2.5 px-4 font-bold text-sky-950 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-sky-600"></span>
                                    AQSM Spatial Misrouting
                                </td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">—</td>
                                <td class="py-2.5 px-3 text-center font-mono font-bold text-sky-900">46.20% (Anchor FG)</td>
                                <td class="py-2.5 px-3 text-center font-mono font-bold text-sky-900">0.2516 (Hung.)</td>
                                <td class="py-2.5 px-3 text-center font-mono text-slate-400">—</td>
                                <td class="py-2.5 px-4 text-right font-mono font-bold text-sky-950">27.32% (40.8%)</td>
                                <td class="py-2.5 px-4 text-sky-950 font-semibold">Adversarial anchors yield modest extra drop ($4.7$ pp) beyond isotropic shift</td>
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
                                <strong>RT-DETR-R18VD:</strong> AQSM degrades mAP@50 monotonically: $10.80\%$ ($\epsilon=4/255$), $8.93\%$ ($\epsilon=8/255$), and $4.05\%$ ($\epsilon=16/255$). Standard Task PGD yields $2.71\%, 1.18\%, 0.35\%$.
                            </p>
                            <p class="text-slate-600 leading-relaxed lang-en">
                                <strong>RT-DETR-R50VD:</strong> AQSM degrades to $23.69\%$ ($4/255$), $15.12\%$ ($8/255$), and $9.14\%$ ($16/255$), showing deeper encoders maintain slightly higher retention.
                            </p>
                        </div>

                        <div class="p-4 rounded-2xl bg-sky-50/70 border border-sky-200/70">
                            <h4 class="font-bold text-sky-950 text-sm mb-2 font-heading">
                                <span class="lang-en">Tesla T4 Stage Profile (38.95 ms, 25.67 FPS)</span>
                                <span class="lang-vi">Phân Rã Tác Vụ Tesla T4 (38.95 ms, 25.67 FPS)</span>
                            </h4>
                            <div class="space-y-1.5 font-mono text-[11px] text-sky-900">
                                <div class="flex justify-between"><span>Backbone (ResNet-18):</span><strong>20.32 ms (52.16%)</strong></div>
                                <div class="flex justify-between"><span>Hybrid Encoder:</span><strong>10.58 ms (27.16%)</strong></div>
                                <div class="flex justify-between"><span>Dynamic Selection Head:</span><strong>1.72 ms (4.40%)</strong></div>
                                <div class="flex justify-between"><span>Deformable Decoder (3L):</span><strong>6.07 ms (15.58%)</strong></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Table 10: Latency Sponge Profiling -->
                <div class="rounded-2xl border border-slate-200 overflow-hidden">
                    <div class="p-4 bg-slate-50/80 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                        <h3 class="text-sm sm:text-base font-bold text-slate-900 font-heading">
                            <span class="lang-en">Table 10: Hardware Execution Latency &amp; Sponge Stress Benchmark</span>
                            <span class="lang-vi">Bảng 10: Đo Đạc Độ Trễ Phần Cứng Thực Tế &amp; Bẫy Trễ Sponge</span>
                        </h3>
                        <span class="text-[11px] font-mono text-slate-500 bg-white px-2.5 py-1 rounded border border-slate-200 w-fit">
                            NVIDIA Tesla T4 &bull; PyTorch FP32 &bull; CUDA Events &bull; 640×640
                        </span>
                    </div>
                    <div class="overflow-x-auto custom-scrollbar">
                        <table class="w-full text-left text-xs text-slate-600">
                            <thead class="bg-slate-100/70 border-b border-slate-200 text-[11px] font-bold text-slate-700 uppercase tracking-wider">
                                <tr>
                                    <th class="py-3 px-4"><span class="lang-en">Model</span><span class="lang-vi">Mô hình</span></th>
                                    <th class="py-3 px-4"><span class="lang-en">Condition</span><span class="lang-vi">Điều kiện</span></th>
                                    <th class="py-3 px-4 text-right"><span class="lang-en">Mean ± Std (ms)</span><span class="lang-vi">Độ Trễ (ms)</span></th>
                                    <th class="py-3 px-4 text-right"><span class="lang-en">Median / p95 (ms)</span><span class="lang-vi">Trung vị / p95</span></th>
                                    <th class="py-3 px-4 text-right"><span class="lang-en">Throughput (FPS)</span><span class="lang-vi">Tốc độ (FPS)</span></th>
                                    <th class="py-3 px-4 text-right"><span class="lang-en">Active Boxes ($\tau \ge 0.35$)</span><span class="lang-vi">Số Hộp Hoạt Động</span></th>
                                    <th class="py-3 px-4"><span class="lang-en">Latency Invariance Status</span><span class="lang-vi">Trạng Thái Đơn Định Độ Trễ</span></th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-100 font-medium text-[11px]">
                                <!-- R18VD -->
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900">RT-DETR-R18VD</td>
                                    <td class="py-2.5 px-4">Clean Baseline</td>
                                    <td class="py-2.5 px-4 text-right font-mono">39.02 ± 0.97</td>
                                    <td class="py-2.5 px-4 text-right font-mono">38.95 / 40.06</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-slate-900">25.63 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono">25.0 boxes</td>
                                    <td class="py-2.5 px-4 text-emerald-700 font-semibold">Nominal baseline reference</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900">RT-DETR-R18VD</td>
                                    <td class="py-2.5 px-4">AQSM Attack ($\epsilon = 16$)</td>
                                    <td class="py-2.5 px-4 text-right font-mono">39.32 ± 0.73</td>
                                    <td class="py-2.5 px-4 text-right font-mono">39.51 / 40.31</td>
                                    <td class="py-2.5 px-4 text-right font-mono text-emerald-700 font-bold">25.43 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono">20.0 boxes</td>
                                    <td class="py-2.5 px-4 text-emerald-700 font-semibold">Invariant ($\Delta t = +0.29$ ms, $+0.75\%$)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900">RT-DETR-R18VD</td>
                                    <td class="py-2.5 px-4">Standard Task PGD</td>
                                    <td class="py-2.5 px-4 text-right font-mono">39.21 ± 0.75</td>
                                    <td class="py-2.5 px-4 text-right font-mono">39.29 / 40.34</td>
                                    <td class="py-2.5 px-4 text-right font-mono text-emerald-700 font-bold">25.50 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono">75.0 boxes</td>
                                    <td class="py-2.5 px-4 text-emerald-700 font-semibold">Invariant ($\Delta t = +0.19$ ms, $+0.48\%$)</td>
                                </tr>
                                <tr class="hover:bg-emerald-50/40 transition bg-emerald-50/20">
                                    <td class="py-2.5 px-4 font-bold text-emerald-950">RT-DETR-R18VD</td>
                                    <td class="py-2.5 px-4 font-semibold text-emerald-900">Proposal-Flooding Sponge Stress</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-emerald-700">38.66 ± 0.76</td>
                                    <td class="py-2.5 px-4 text-right font-mono">38.73 / 39.92</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-emerald-700">25.87 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-rose-600">207.0 boxes</td>
                                    <td class="py-2.5 px-4 text-emerald-800 font-bold">Immune ($O(1)$ fixed $N_q=300$ decoding)</td>
                                </tr>

                                <!-- R50VD -->
                                <tr class="hover:bg-slate-50/60 transition bg-slate-50/30">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900">RT-DETR-R50VD</td>
                                    <td class="py-2.5 px-4">Clean Baseline</td>
                                    <td class="py-2.5 px-4 text-right font-mono">74.89 ± 1.36</td>
                                    <td class="py-2.5 px-4 text-right font-mono">74.76 / 76.38</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-slate-900">13.35 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono">21.0 boxes</td>
                                    <td class="py-2.5 px-4 text-emerald-700 font-semibold">Nominal baseline reference</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-slate-50/30">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900">RT-DETR-R50VD</td>
                                    <td class="py-2.5 px-4">AQSM Attack ($\epsilon = 16$)</td>
                                    <td class="py-2.5 px-4 text-right font-mono">74.41 ± 1.46</td>
                                    <td class="py-2.5 px-4 text-right font-mono">74.23 / 75.83</td>
                                    <td class="py-2.5 px-4 text-right font-mono text-emerald-700 font-bold">13.44 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono">5.0 boxes</td>
                                    <td class="py-2.5 px-4 text-emerald-700 font-semibold">Invariant ($\Delta t = -0.48$ ms)</td>
                                </tr>
                                <tr class="hover:bg-emerald-50/40 transition bg-emerald-50/20">
                                    <td class="py-2.5 px-4 font-bold text-emerald-950">RT-DETR-R50VD</td>
                                    <td class="py-2.5 px-4 font-semibold text-emerald-900">Proposal-Flooding Sponge Stress</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-emerald-700">72.76 ± 1.35</td>
                                    <td class="py-2.5 px-4 text-right font-mono">72.58 / 74.63</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-emerald-700">13.74 FPS</td>
                                    <td class="py-2.5 px-4 text-right font-mono font-bold text-rose-600">136.0 boxes</td>
                                    <td class="py-2.5 px-4 text-emerald-800 font-bold">Immune ($O(1)$ fixed $N_q=300$ decoding)</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>

        <!-- Multi-Seed Stability and Cross-Architecture Comparison (Table 13) -->
        <section id="generalization" class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 mb-20">
            <div class="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-sm">
                <div class="mb-6 text-center max-w-3xl mx-auto">
                    <span class="text-xs font-bold text-sky-600 uppercase tracking-wider">
                        <span class="lang-en">Cross-Depth &amp; 5-Seed Generalization</span>
                        <span class="lang-vi">Đối Chiếu Chiều Sâu &amp; Độ Ổn Định 5 Hạt Giống</span>
                    </span>
                    <h2 class="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1 mb-3 font-heading">
                        <span class="lang-en">Table 13: Consolidated Benchmark — RT-DETR-R18VD vs. RT-DETR-R50VD</span>
                        <span class="lang-vi">Bảng 13: Bảng Hợp Nhất Thực Nghiệm — RT-DETR-R18VD vs. RT-DETR-R50VD</span>
                    </h2>
                    <p class="text-sm text-slate-600 lang-en">
                        Direct symmetric comparison between 3-layer RT-DETR-R18VD and 6-layer RT-DETR-R50VD on COCO val2017 ($N = 300$, $640 \times 640$, $\epsilon = 16/255$). Across 5 independent random seeds ($\{42, 100, 2024, 777, 9999\}$), AQSM achieves $\text{ICC}(2,1) = 0.946$ (R18VD) and $0.908$ (R50VD), confirming architectural stability.
                    </p>
                    <p class="text-sm text-slate-600 lang-vi">
                        So sánh đối xứng trực tiếp giữa RT-DETR-R18VD (3 tầng decoder) và RT-DETR-R50VD (6 tầng decoder) trên COCO val2017 ($N = 300$, $640 \times 640$, $\epsilon = 16/255$). Qua 5 hạt giống ngẫu nhiên độc lập ($\{42, 100, 2024, 777, 9999\}$), đòn đánh AQSM đạt hệ số tương quan nội lớp $\text{ICC}(2,1) = 0.946$ (R18VD) và $0.908$ (R50VD), khẳng định tính ổn định cấu trúc.
                    </p>
                </div>

                <div class="rounded-2xl border border-slate-200 overflow-hidden mb-4">
                    <div class="p-4 bg-slate-50/80 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                        <span class="text-xs font-semibold text-slate-700">
                            <span class="lang-en">Unified Diagnostic Axis Across 12 Dimensions</span>
                            <span class="lang-vi">Trục Chẩn Đoán Hợp Nhất Qua 12 Chiều Đánh Giá</span>
                        </span>
                        <span class="text-[11px] font-mono text-sky-700 bg-sky-50 px-2.5 py-1 rounded border border-sky-200 w-fit">
                            ResNet-18 (20M, 3 Dec.) vs. ResNet-50 (42M, 6 Dec.)
                        </span>
                    </div>
                    <div class="overflow-x-auto custom-scrollbar">
                        <table class="w-full text-left text-xs text-slate-600">
                            <thead class="bg-slate-100/70 border-b border-slate-200 text-[11px] font-bold text-slate-700 uppercase tracking-wider">
                                <tr>
                                    <th class="py-3 px-5"><span class="lang-en">Diagnostic Axis ($\epsilon = 16/255$ unless noted)</span><span class="lang-vi">Trục Chẩn Đoán ($\epsilon = 16/255$ trừ khi ghi rõ)</span></th>
                                    <th class="py-3 px-4 text-center">RT-DETR-R18VD (3 dec., $N=300$)</th>
                                    <th class="py-3 px-5 text-center">RT-DETR-R50VD (6 dec., $N=300$)</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-100 font-medium text-[11px]">
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Nominal Clean mAP@50 (mAP@50:95)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">66.93% (50.24%)</td>
                                    <td class="py-2.5 px-5 text-center font-mono font-bold text-slate-900">72.95% (55.73%)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-rose-50/20">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">AQSM Cached Seed 42 mAP@50 [Rel. Drop]</td>
                                    <td class="py-2.5 px-4 text-center font-mono text-rose-600 font-semibold">4.05% [−93.96%]</td>
                                    <td class="py-2.5 px-5 text-center font-mono text-rose-600 font-semibold">9.14% [−87.47%]</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-rose-50/30">
                                    <td class="py-2.5 px-5 font-bold text-rose-950">AQSM 5-Seed Fresh Mean ± Std [Rel. Drop]</td>
                                    <td class="py-2.5 px-4 text-center font-mono font-bold text-rose-700">4.00 ± 0.74% [−94.02%]</td>
                                    <td class="py-2.5 px-5 text-center font-mono font-bold text-rose-700">10.19 ± 1.10% [−86.03%]</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">AQSM 5-Seed Hungarian Drift / Anchor FG Hit</td>
                                    <td class="py-2.5 px-4 text-center font-mono">0.2345 ± 0.0037 / 47.12 ± 0.48%</td>
                                    <td class="py-2.5 px-5 text-center font-mono">0.2237 ± 0.0010 / 50.09 ± 0.45%</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-sky-50/30">
                                    <td class="py-2.5 px-5 font-semibold text-sky-950">AQSM 5-Seed Inter-Seed Consistency ($\text{ICC}(2,1)$)</td>
                                    <td class="py-2.5 px-4 text-center font-mono font-bold text-sky-800">ICC = 0.946 ($p \le 2.32 \times 10^{-49}$)</td>
                                    <td class="py-2.5 px-5 text-center font-mono font-bold text-sky-800">ICC = 0.908 ($p \le 1.51 \times 10^{-34}$)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">AQSM Budget Sweep ($\epsilon = 4/255, 8/255$)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">10.80%, 8.93%</td>
                                    <td class="py-2.5 px-5 text-center font-mono">23.69%, 15.12%</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">PGD Vanishing / Standard Task PGD mAP@50</td>
                                    <td class="py-2.5 px-4 text-center font-mono">1.32% / 0.35%</td>
                                    <td class="py-2.5 px-5 text-center font-mono">5.19% / 2.76%</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Causal Surgery: $\text{C}_{\text{full}}$ / $\text{D}_{\text{full}}$ / $\text{D}_{\text{index}}$</td>
                                    <td class="py-2.5 px-4 text-center font-mono">9.31% / 30.37% / 39.11% (58.4% ret.)</td>
                                    <td class="py-2.5 px-5 text-center font-mono">14.43% / 42.75% / 53.57% (73.4% ret.)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-sky-50/40">
                                    <td class="py-2.5 px-5 font-bold text-sky-950">Shapley Share (Per-Image $AP_i$: Mem / Query)</td>
                                    <td class="py-2.5 px-4 text-center font-mono font-bold text-sky-900">60.59% / 39.41% [pooled 66.75%]</td>
                                    <td class="py-2.5 px-5 text-center font-mono font-bold text-sky-900">68.16% / 31.84% [pooled 72.19%]</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Encoder Memory Cosine / Top-300 Overlap / Top-15 Evict</td>
                                    <td class="py-2.5 px-4 text-center font-mono">0.6140 / 16.11% / 97.22%</td>
                                    <td class="py-2.5 px-5 text-center font-mono">0.6758 / 21.40% / 91.76%</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Deformable Attention Entropy ($H_{\mathrm{all}}$ Clean $\to$ AQSM)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">0.8032 $\to$ 0.7938 ($H_{L_1}: 0.8287 \to 0.8185$)</td>
                                    <td class="py-2.5 px-5 text-center font-mono">0.8156 $\to$ 0.8071 ($H_{L_1}: 0.8444 \to 0.8365$)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Deformable Sampling FG Hit Rate (Clean $\to$ AQSM)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">63.53% $\to$ 44.17% ($d = -0.86$)</td>
                                    <td class="py-2.5 px-5 text-center font-mono">63.32% $\to$ 48.31% ($d = -0.68$)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Controlled Misrouting (5s Reassign / 5s Shift / AQSM)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">63.21 ± 0.47% / 32.06 ± 0.82% / 27.32%</td>
                                    <td class="py-2.5 px-5 text-center font-mono">72.36 ± 0.16% / 47.89 ± 0.67% / 42.34%</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">ESM AQSM TPR (AUROC) / Adaptive Raw Bypass</td>
                                    <td class="py-2.5 px-4 text-center font-mono">47.62% (0.7881) / 94.76–97.14%</td>
                                    <td class="py-2.5 px-5 text-center font-mono">25.71% (0.6914) / 93.33–97.14%</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Zero-Shot SQA mAP@50 (Clean $\to$ SQA; AQSM Dyn $\to$ SQA)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">66.93% $\to$ 66.98% / 4.05% $\to$ 3.97%</td>
                                    <td class="py-2.5 px-5 text-center font-mono">72.95% $\to$ 72.97% / 9.14% $\to$ 9.25%</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-5 font-semibold text-slate-900">Tesla T4 Latency (Query Selection Share %)</td>
                                    <td class="py-2.5 px-4 text-center font-mono">38.95 ms (4.40%, 25.67 FPS)</td>
                                    <td class="py-2.5 px-5 text-center font-mono">73.83 ms (2.31%, 13.54 FPS)</td>
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
                                ESM flags stationary AQSM at <strong>47.62% TPR</strong> ($\text{AUROC} = 0.7881$) on R18VD ($25.71\%$ on R50VD) at $10\%$ FPR. The two-sided Active-Box Watchdog detects extreme Tier B attacks strongly: <strong>85.24%</strong> Vanishing and <strong>89.05%</strong> Task PGD.
                            </p>
                            <p class="text-slate-600 lang-vi mb-2">
                                ESM phát hiện AQSM tĩnh đạt <strong>47.62% TPR</strong> ($\text{AUROC} = 0.7881$) trên R18VD ($25.71\%$ trên R50VD) ở mức FPR $10\%$. Bộ giám sát số hộp phát hiện tốt các đòn đánh cực đoan: <strong>85.24%</strong> Vanishing và <strong>89.05%</strong> Task PGD.
                            </p>
                        </div>
                        <span class="text-[10px] font-mono text-slate-400 bg-white px-2 py-0.5 rounded border border-slate-200 w-fit">Table 11 &bull; N_cal=90, N_test=210</span>
                    </div>

                    <div class="p-4 rounded-2xl bg-slate-50 border border-slate-200/70 flex flex-col justify-between">
                        <div>
                            <div class="flex items-center gap-2 mb-1.5 font-bold text-slate-900">
                                <span class="w-5 h-5 rounded-md bg-amber-100 text-amber-600 flex items-center justify-center text-[10px] font-bold">(b)</span>
                                <span class="lang-en">White-Box Adaptive Evasion</span>
                                <span class="lang-vi">Qua Mặt Thích Nghi White-Box</span>
                            </div>
                            <p class="text-slate-600 lang-en mb-2">
                                A Carlini-Tramèr white-box adversary suppresses ESM detection TPR down to <strong>2.86%–5.24%</strong> (<strong>94.76%–97.14% raw bypass</strong>), while still severely collapsing detection accuracy to $8.91\%–11.90\%$ mAP@50.
                            </p>
                            <p class="text-slate-600 lang-vi mb-2">
                                Đối thủ thích nghi white-box Carlini-Tramèr dồn TPR của ESM xuống chỉ còn <strong>2.86%–5.24%</strong> (<strong>tỷ lệ qua mặt 94.76%–97.14%</strong>), trong khi vẫn làm sụp đổ độ chính xác phát hiện về mức $8.91\%–11.90\%$ mAP@50.
                            </p>
                        </div>
                        <span class="text-[10px] font-mono text-slate-400 bg-white px-2 py-0.5 rounded border border-slate-200 w-fit">Adaptive Loss &bull; λ ∈ {0.2, 0.5, 1.0, 2.0}</span>
                    </div>

                    <div class="p-4 rounded-2xl bg-sky-50/70 border border-sky-200/70 flex flex-col justify-between">
                        <div>
                            <div class="flex items-center gap-2 mb-1.5 font-bold text-sky-950">
                                <span class="w-5 h-5 rounded-md bg-sky-200 text-sky-800 flex items-center justify-center text-[10px] font-bold">(c)</span>
                                <span class="lang-en">SQA Empirical Null Result</span>
                                <span class="lang-vi">Kết Quả Null Thực Nghiệm Của SQA</span>
                            </div>
                            <p class="text-sky-900 lang-en mb-2">
                                Live GPU evaluation of zero-shot SQA yields a null result: $4.05\% \to 3.97\%$ ($-0.08$ pp) on R18VD, and $9.14\% \to 9.25\%$ ($+0.11$ pp) on R50VD, because corrupted encoder memory ($M_{\mathrm{adv}}$) cannot be decoded by unhardened queries.
                            </p>
                            <p class="text-sky-900 lang-vi mb-2">
                                Đo đạc trực tiếp trên GPU cho thấy SQA zero-shot cho kết quả null: $4.05\% \to 3.97\%$ ($-0.08$ pp) trên R18VD, và $9.14\% \to 9.25\%$ ($+0.11$ pp) trên R50VD, vì bộ nhớ encoder bị phá hoại ($M_{\mathrm{adv}}$) không thể giải mã bởi query chưa gia cố.
                            </p>
                        </div>
                        <span class="text-[10px] font-mono text-sky-700 bg-sky-100/80 px-2 py-0.5 rounded border border-sky-300 w-fit">Table 12 &bull; 300 Dyn + 30 Static</span>
                    </div>
                </div>

                <!-- Table 12: Architectural Hardening via SQA Validation & Controls -->
                <div class="rounded-2xl border border-slate-200 overflow-hidden">
                    <div class="p-4 sm:p-5 bg-slate-50/80 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                        <div>
                            <h3 class="text-sm sm:text-base font-bold text-slate-900 font-heading">
                                <span class="lang-en">Table 12: Zero-Shot Static Query Augmentation (SQA) &amp; Capacity Controls</span>
                                <span class="lang-vi">Bảng 12: Đánh Giá SQA Zero-Shot &amp; Các Nhóm Đối Chứng Dung Lượng</span>
                            </h3>
                            <p class="text-xs text-slate-500 mt-0.5 lang-en">COCO val2017 ($N = 300$, native $640 \times 640$, adaptive subset $N_{\mathrm{adapt}} = 20$) across RT-DETR-R18VD and RT-DETR-R50VD.</p>
                            <p class="text-xs text-slate-500 mt-0.5 lang-vi">COCO val2017 ($N = 300$, độ phân giải $640 \times 640$, tập con thích nghi $N_{\mathrm{adapt}} = 20$) trên RT-DETR-R18VD và RT-DETR-R50VD.</p>
                        </div>
                        <span class="text-[11px] font-mono text-slate-500 bg-white px-2.5 py-1 rounded border border-slate-200 w-fit">
                            $N_{\mathrm{static}} = 30$ &bull; Uniform Grid &bull; Tesla T4
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
                                    <th class="py-3 px-3 text-center">T4 Latency</th>
                                    <th class="py-3 px-4"><span class="lang-en">Mechanistic Finding</span><span class="lang-vi">Phát Hiện Cơ Chế</span></th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-100 font-medium text-[11px]">
                                <!-- R18VD -->
                                <tr class="bg-slate-100/60">
                                    <td colspan="6" class="py-1.5 px-4 font-bold text-slate-800 text-[10px] uppercase tracking-wider">
                                        RT-DETR-R18VD (3 Decoder Layers)
                                    </td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900 pl-6">Clean Dynamic-300 (Ref.)</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-slate-900">66.93%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">50.24%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">86.48%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">25.37 ms</td>
                                    <td class="py-2.5 px-4 text-slate-500">Nominal clean reference baseline</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900 pl-6">Clean SQA Hardened (300 dyn. + 30 static)</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-emerald-700">66.98%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">50.25%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">86.39%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">25.48 ms</td>
                                    <td class="py-2.5 px-4 text-emerald-700">Zero clean penalty (+0.05 pp, +0.43% latency)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-rose-50/20">
                                    <td class="py-2.5 px-4 font-semibold text-rose-950 pl-6">AQSM (ε = 16): Dynamic-300 Baseline</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-rose-600">4.05%</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-rose-600">2.10%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">17.90%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">25.37 ms</td>
                                    <td class="py-2.5 px-4 text-rose-600">Catastrophic collapse (6.05% retention)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 text-slate-700 pl-6">Control 1: Dynamic Top-330</td>
                                    <td class="py-2.5 px-3 text-right font-mono">4.06%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">2.10%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">17.95%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">25.48 ms</td>
                                    <td class="py-2.5 px-4 text-slate-500">Expanding dynamic pool fails (+0.01 pp)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 text-slate-700 pl-6">Control 2: Random-30 Static</td>
                                    <td class="py-2.5 px-3 text-right font-mono">4.03%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">2.04%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">18.04%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">25.48 ms</td>
                                    <td class="py-2.5 px-4 text-slate-500">Random static anchors yield no recovery (−0.02 pp)</td>
                                </tr>
                                <tr class="hover:bg-sky-50/40 transition bg-sky-50/30">
                                    <td class="py-2.5 px-4 font-bold text-sky-950 pl-6">AQSM SQA (Grid-30 Static) [Proposed]</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-sky-700">3.97%</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-sky-700">2.03%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">18.04%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">25.48 ms</td>
                                    <td class="py-2.5 px-4 text-sky-900 font-semibold">Null recovery (−0.08 pp); corrupted memory $M_{\mathrm{adv}}$ dominates</td>
                                </tr>
                                <tr class="hover:bg-amber-50/40 transition">
                                    <td class="py-2.5 px-4 text-slate-700 pl-6">Adaptive Joint Evasion on SQA ($N_{\mathrm{adapt}} = 20$)</td>
                                    <td class="py-2.5 px-3 text-right font-mono">6.93%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">4.35%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">8.45%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">25.48 ms</td>
                                    <td class="py-2.5 px-4 text-amber-800">89.65% bypass on evaluated $N=20$ adaptive subset</td>
                                </tr>

                                <!-- R50VD -->
                                <tr class="bg-slate-100/60">
                                    <td colspan="6" class="py-1.5 px-4 font-bold text-slate-800 text-[10px] uppercase tracking-wider">
                                        RT-DETR-R50VD (6 Decoder Layers)
                                    </td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900 pl-6">Clean Dynamic-300 (Ref.)</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-slate-900">72.95%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">55.73%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">90.58%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">68.09 ms</td>
                                    <td class="py-2.5 px-4 text-slate-500">Nominal clean reference baseline</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition">
                                    <td class="py-2.5 px-4 font-semibold text-slate-900 pl-6">Clean SQA Hardened (300 dyn. + 30 static)</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-emerald-700">72.97%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">55.73%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">90.57%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">68.85 ms</td>
                                    <td class="py-2.5 px-4 text-emerald-700">Zero clean penalty (+0.02 pp, +1.12% latency)</td>
                                </tr>
                                <tr class="hover:bg-slate-50/60 transition bg-rose-50/20">
                                    <td class="py-2.5 px-4 font-semibold text-rose-950 pl-6">AQSM (ε = 16): Dynamic-300 Baseline</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-rose-600">9.14%</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-rose-600">5.41%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">27.99%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">68.09 ms</td>
                                    <td class="py-2.5 px-4 text-rose-600">Unhardened baseline (12.53% retention)</td>
                                </tr>
                                <tr class="hover:bg-sky-50/40 transition bg-sky-50/30">
                                    <td class="py-2.5 px-4 font-bold text-sky-950 pl-6">AQSM SQA (Grid-30 Static) [Proposed]</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-sky-700">9.25%</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-sky-700">5.52%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">28.08%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">68.85 ms</td>
                                    <td class="py-2.5 px-4 text-sky-900 font-semibold">Null recovery (+0.11 pp); corrupted memory $M_{\mathrm{adv}}$ dominates</td>
                                </tr>
                                <tr class="hover:bg-amber-50/40 transition">
                                    <td class="py-2.5 px-4 text-slate-700 pl-6">Adaptive Joint Evasion on SQA ($N_{\mathrm{adapt}} = 20$)</td>
                                    <td class="py-2.5 px-3 text-right font-mono">15.04%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">9.81%</td>
                                    <td class="py-2.5 px-3 text-right font-mono">16.43%</td>
                                    <td class="py-2.5 px-3 text-center font-mono">68.85 ms</td>
                                    <td class="py-2.5 px-4 text-amber-800">79.38% bypass on evaluated $N=20$ adaptive subset</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <div class="p-3.5 bg-slate-50/80 border-t border-slate-200 text-[11px] text-slate-500 leading-relaxed">
                        <span class="lang-en"><strong>Mechanistic Lesson:</strong> Zero-shot SQA reveals an informative empirical null recovery result ($4.05\% \to 3.97\%$ on R18VD; $9.14\% \to 9.25\%$ on R50VD). Consistent with causal surgery (where Arm $\text{C}_{\text{full}}$ retains only $9.31\%$ / $14.43\%$ even with all 300 clean queries restored), injecting unhardened static queries cannot compensate for severely corrupted encoder memory representations ($M_{\mathrm{adv}}$). Defending dynamic-query detectors requires coupled multi-scale encoder representation hardening alongside coordinate-calibrated static priors.</span>
                        <span class="lang-vi"><strong>Bài học Cơ chế:</strong> Thử nghiệm SQA zero-shot đem lại kết quả null thực nghiệm mang tính chẩn đoán sâu sắc ($4.05\% \to 3.97\%$ trên R18VD; $9.14\% \to 9.25\%$ trên R50VD). Phù hợp với phân tích hoán đổi nhân quả (trong đó Nhánh $\text{C}_{\text{full}}$ chỉ giữ được $9.31\%$ / $14.43\%$ ngay cả khi khôi phục toàn bộ 300 query sạch), việc bổ sung query tĩnh chưa gia cố không thể bù đắp cho biểu diễn bộ nhớ encoder đã bị tàn phá nặng ($M_{\mathrm{adv}}$). Việc phòng thủ các detector truy vấn động đòi hỏi phải kết hợp gia cố biểu diễn encoder đa tầng cùng với các anchor tĩnh được chuẩn hóa tọa độ.</span>
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
            mean_en: 'Contains semantic contextual representations. Perturbing M accounts for 60.59% (R18VD) and 68.16% (R50VD) of detector collapse.',
            mean_vi: 'Chứa biểu diễn ngữ nghĩa của cảnh vật. Làm sai lệch M chiếm tới 60.59% (R18VD) và 68.16% (R50VD) nguyên nhân gây sụp đổ mô hình.'
        },
        'query': {
            title_en: 'Query Prior (Q)',
            title_vi: 'Query Prior (Vị trí Tiên nghiệm Q)',
            calc_en: 'Top-300 candidate positions and reference coordinates routed by dynamic query selection head.',
            calc_vi: 'Top-300 vị trí và tọa độ anchor ứng viên được lựa chọn bởi tầng dynamic query selection.',
            mean_en: 'Clean: foreground object centers. Adversarial: anchors misrouted into background noise clusters (39.41% share on R18VD, 31.84% on R50VD).',
            mean_vi: 'Sạch: anchor bao phủ đúng tâm đối tượng. Đối kháng: anchor bị định tuyến sai lệch vào các cụm nền nhiễu (39.41% trên R18VD, 31.84% trên R50VD).'
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
            calc_en: 'H(A) = - Σ a_k log(a_k) / log(K), computed over K=4 deformable sampling offsets across all heads and layers.',
            calc_vi: 'H(A) = - Σ a_k log(a_k) / log(K), tính toán trên K=4 điểm lấy mẫu biến dạng qua tất cả các đầu và tầng decoder.',
            mean_en: 'Measures sampling weight concentration. Under AQSM, H(A) stays virtually constant (0.8032 -> 0.7938 on R18VD, d = -0.19), refuting attention scattering.',
            mean_vi: 'Đo lường mức độ tập trung trọng số lấy mẫu. Dưới AQSM, H(A) gần như không đổi (0.8032 -> 0.7938 trên R18VD, d = -0.19), bác bỏ giả thuyết tán xạ chú ý.'
        },
        'sampling_hit': {
            title_en: 'Spatial Sampling Hit Rate (FG %)',
            title_vi: 'Tỷ lệ Lấy mẫu trúng Foreground (FG %)',
            calc_en: 'Percentage of deformable sampling point locations that fall inside ground-truth foreground object masks.',
            calc_vi: 'Tỷ lệ phần trăm các điểm lấy mẫu biến dạng rơi trúng bên trong vùng đối tượng foreground ground-truth.',
            mean_en: 'Collapses from 63.53% (clean) down to 44.17% under AQSM on R18VD (p = 2.54 × 10^-32), starving cross-attention heads of genuine object evidence.',
            mean_vi: 'Sụp đổ từ 63.53% (ảnh sạch) xuống 44.17% dưới AQSM trên R18VD (p = 2.54 × 10^-32), khiến các đầu cross-attention suy giảm nghiêm trọng tín hiệu đặc trưng đối tượng thật.'
        },
        'ref_drift': {
            title_en: 'Reference Coordinate Drift (Δp)',
            title_vi: 'Độ trôi Tọa độ Tham chiếu (Δp)',
            calc_en: 'Mean displacement ||p_adv - p_clean||_2 of anchor centers, alongside permutation-invariant Hungarian set distance.',
            calc_vi: 'Độ dịch chuyển tâm anchor ||p_adv - p_clean||_2 kết hợp khoảng cách tập hợp bất biến hoán vị Hungarian.',
            mean_en: 'Quantifies spatial diversion into peripheral background noise (Clean: 0.0000, AQSM: 0.4750 R18VD / 0.4503 R50VD).',
            mean_vi: 'Định lượng mức độ trôi lệch không gian vào nhiễu nền ngoại vi (Clean: 0.0000, AQSM: 0.4750 R18VD / 0.4503 R50VD).'
        },
        'dropoff_ratio': {
            title_en: 'Proposal Dropoff Ratio D(X)',
            title_vi: 'Tỷ số Suy giảm Đề xuất D(X)',
            calc_en: 'D(X) = (s_(1) + 10^-4) / (s_(300) + 10^-4), ratio of highest proposal logit to the 300th boundary candidate.',
            calc_vi: 'D(X) = (s_(1) + 10^-4) / (s_(300) + 10^-4), tỷ số giữa logit đề xuất cao nhất và đề xuất thứ 300 tại biên.',
            mean_en: 'Measures confidence distribution sharpness. Clean inference produces steep decay (34.73), while AQSM flattens it (10.54).',
            mean_vi: 'Đo độ dốc phân phối độ tin cậy. Ảnh sạch có phân phối rất dốc (34.73), trong khi AQSM san phẳng phân phối này (10.54).'
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
