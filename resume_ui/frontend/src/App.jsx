import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import {
    LayoutDashboard, Briefcase, Users, BarChart2, Upload, X, Check, Plus,
    FileText, TrendingUp, Award, Mail, ChevronRight,
    Sparkles, Clock, Star, AlertCircle, CheckCircle, Zap, ArrowRight,
    RefreshCw, Sun, Moon, Edit3, Save, Activity, Target, Download,
    Menu, ChevronDown, Info, Phone, Globe, ExternalLink,
} from "lucide-react";
import {
    Tooltip, ResponsiveContainer,
    RadarChart, PolarGrid, PolarAngleAxis, Radar,
} from "recharts";

// ── Theme palettes ─────────────────────────────────────────────────────────
// Pure inline-style design tokens (no Tailwind). Switch via the `dark` state
// in App. All component colours read from the module-level `C` ref below.
const DARK = {
    bg: "#09090b", surface: "rgba(255,255,255,0.04)", border: "rgba(255,255,255,0.09)",
    text: "#f4f4f5", sub: "#71717a", muted: "#52525b",
    blue: "#6263c6", green: "#22c55e", amber: "#f59e0b", pink: "#ec4899", teal: "#14b8a6",
    sidebarBg: "rgba(255,255,255,0.02)", topbarBg: "rgba(9,9,11,0.90)",
    inputBg: "rgba(255,255,255,0.05)", cardText: "#d4d4d8",
    drawerBg: "#0f0f11", scrollThumb: "rgba(255,255,255,0.09)",
    chartTooltip: "#0f0f11", tableFocus: "rgba(99,102,241,0.06)",
    chartSkill: "#818cf8", chartSemantic: "#34d399", chartFinal: "#fbbf24",
};
const LIGHT = {
    // Page & surfaces
    bg:       "#f0f2f5",
    surface:  "#ffffff",
    border:   "rgba(0,0,0,0.14)",

    // Text hierarchy
    text:     "#111827",
    sub:      "#374151",
    muted:    "#6b7280",
    cardText: "#1f2937",

    // Accent colours
    blue:     "#4f46e5",
    green:    "#16a34a",
    amber:    "#d97706",
    pink:     "#db2777",
    teal:     "#0d9488",

    // Chrome
    sidebarBg:  "#f8fafc",
    topbarBg:   "rgba(255,255,255,0.92)",
    drawerBg:   "#ffffff",

    // Inputs
    inputBg:    "rgba(0,0,0,0.08)",
    tableFocus: "rgba(79,70,229,0.06)",
    scrollThumb: "rgba(0,0,0,0.18)",

    // Charts — darker shades so they read on white backgrounds
    chartTooltip: "#ffffff",
    chartSkill:   "#4f46e5",
    chartSemantic:"#0d9488",
    chartFinal:   "#d97706",
};

// `C` is a module-level mutable ref that is synchronously reassigned at the
// start of every App render cycle (before any child reads it), so all
// components always read the current theme without prop-drilling.
let C = DARK;

// Per-model classification thresholds matching scoring_engine.py
const MODEL_THRESH = { mpnet: 0.45, mxbai: 0.55, arctic: 0.50 };

// Base card style helper with optional overrides
const card = (extra = {}) => ({
    background: C.surface, border: `1px solid ${C.border}`, borderRadius: 16, padding: 24, ...extra,
});

// ── Backend URL — must match Flask CORS config and be updated for production deployment
const BASE = "http://localhost:5001";

const PIPELINE_STEPS = (skillW = 55, modelName = "MPNet") => [
    { label: "resume_parser.py", desc: "Extracting text from PDFs via pdfplumber" },
    { label: "information_extractor.py", desc: "Regex + text rules — names, email, phone, skills, experience" },
    { label: "semantic_matcher.py", desc: `Generating sentence embeddings using ${modelName}` },
    { label: "semantic_matcher.py", desc: "Computing cosine similarity vs. job description" },
    { label: "scoring_engine.py", desc: "Applying eligibility filter (min experience check)" },
    { label: "scoring_engine.py", desc: `Weighted ranking — ${skillW}% skill / ${100 - skillW}% semantic` },
];

// Embedding model definitions — must match VALID_MODELS in app.py
const MODELS = {
    mpnet: {
        key: "mpnet",
        name: "MPNet",
        short: "multi-qa-mpnet-base-dot-v1",
        badge: "Balanced performance",
        color: "#818cf8",
        desc: "Sentence-BERT (SBERT) model trained on 215M question-answer pairs using a Siamese network — the only true SBERT model in this system and the most semantically aware for resume matching.",
        detail: "768-dim · SBERT architecture · dot-product similarity · best for demos & high accuracy.",
    },
    mxbai: {
        key: "mxbai",
        name: "MxBai",
        short: "mixedbread-ai/mxbai-embed-large-v1",
        badge: "2025 MTEB #1",
        color: "#e879f9",
        desc: "State-of-the-art embedding model by Mixedbread AI — ranked #1 on the MTEB English leaderboard. Uses contrastive training for superior retrieval and ranking performance.",
        detail: "1024-dim · 335M params · contrastive training · highest real-world retrieval accuracy.",
    },
    arctic: {
        key: "arctic",
        name: "Arctic",
        short: "Snowflake/snowflake-arctic-embed-m-v1.5",
        badge: "Optimized ranking",
        color: "#64d1ff",
        desc: "Snowflake's enterprise embedding model — not SBERT, but a dedicated retrieval model optimised for high-precision query – document ranking like job description to resume matching and search at scale.",
        detail: "768-dim · MTEB top retrieval · enterprise-grade · ideal for high-precision screening.",
    },
};

function useWindowWidth() {
    const [w, setW] = useState(typeof window !== "undefined" ? window.innerWidth : 1200);
    useEffect(() => {
        const h = () => setW(window.innerWidth);
        window.addEventListener("resize", h);
        return () => window.removeEventListener("resize", h);
    }, []);
    return w;
}

const CircularRing = ({ value, size = 52, sw = 4.5, color }) => {
    const clr = color || C.blue;
    const safeVal = Math.min(100, Math.max(0, value || 0));
    const r = (size - sw) / 2;
    const circ = r * 2 * Math.PI;
    const offset = circ - (safeVal / 100) * circ;
    return (
        <div style={{ position: "relative", width: size, height: size, flexShrink: 0 }}>
            <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
                <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(128,128,128,0.13)" strokeWidth={sw} />
                <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={clr} strokeWidth={sw}
                    strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
                    style={{ transition: "stroke-dashoffset 1s cubic-bezier(.4,0,.2,1)" }} />
            </svg>
            <span style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: size * 0.20, fontWeight: 700, color: C.text, fontVariantNumeric: "tabular-nums" }}>
                {safeVal}%
            </span>
        </div>
    );
};

const GaugeArc = ({ value }) => {
    const r = 64, sw = 12, cx = 85, cy = 82;
    const circ = Math.PI * r;
    const off = circ - (value / 100) * circ;
    const col = value >= 80 ? C.green : value >= 55 ? C.blue : C.amber;
    // Layout:
    //   Container is 88px tall (matches SVG height).
    //   Arc top: y=18  Arc mouth inner edge: y=76  Arc outer bottom: y=88
    //   Text is absolutely placed with bottom=14, so the block's bottom sits at
    //   y=74 (2px above inner arc edge).  This puts the text visually INSIDE the
    //   bowl, right at its lower opening — not floating below the arc.
    return (
        <div style={{ position: "relative", width: 170, flexShrink: 0 }}>
            <svg width={170} height={88} viewBox="0 0 170 88" style={{ display: "block" }}>
                {/* Track */}
                <path d={`M${cx - r} ${cy} A${r} ${r} 0 0 1 ${cx + r} ${cy}`}
                    fill="none" stroke="rgba(128,128,128,0.11)" strokeWidth={sw} strokeLinecap="round" />
                {/* Filled arc */}
                <path d={`M${cx - r} ${cy} A${r} ${r} 0 0 1 ${cx + r} ${cy}`}
                    fill="none" stroke={col} strokeWidth={sw} strokeLinecap="round"
                    strokeDasharray={circ} strokeDashoffset={off}
                    style={{ transition: "stroke-dashoffset 1.2s ease", filter: `drop-shadow(0 0 7px ${col}88)` }} />
            </svg>
            {/* Text overlay — absolutely placed inside the arc bowl */}
            <div style={{
                position: "absolute",
                bottom: 4,
                left: 0, right: 0,
                textAlign: "center",
                lineHeight: 1,
                pointerEvents: "none",
            }}>
                <div style={{
                    fontSize: 28, fontWeight: 900, color: col,
                    fontVariantNumeric: "tabular-nums", letterSpacing: "-0.02em",
                }}>{value}%</div>
                <div style={{
                    fontSize: 10, color: C.sub, marginTop: 5,
                    fontWeight: 600, letterSpacing: "0.04em",
                }}>Final Score</div>
            </div>
        </div>
    );
};

const Badge = ({ status }) => {
    const m = {
        Shortlisted: { bg: "rgba(34,197,94,.12)", col: "#22c55e" },
        Rejected: { bg: "rgba(239,68,68,.12)", col: "#ef4444" },
        Reviewing: { bg: "rgba(245,158,11,.12)", col: "#f59e0b" },
    };
    const s = m[status] || m.Reviewing;
    return (
        <span style={{ padding: "3px 10px", borderRadius: 20, fontSize: 11, fontWeight: 600, background: s.bg, color: s.col, border: `1px solid ${s.col}40`, display: "inline-flex", alignItems: "center", gap: 5, lineHeight: 1.2 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: s.col, flexShrink: 0 }} />{status}
        </span>
    );
};

const SkillChip = ({ label, matched }) => (
    <span style={{
        padding: "4px 10px", borderRadius: 6, fontSize: 11, fontWeight: 500,
        background: matched ? `${C.blue}22` : C.inputBg,
        color: matched ? C.blue : C.sub,
        border: `1px solid ${matched ? `${C.blue}44` : C.border}`,
        display: "inline-flex", alignItems: "center", gap: 4, lineHeight: 1,
    }}>
        {matched ? <Check size={10} strokeWidth={2.5} /> : <X size={10} />} {label}
    </span>
);

const Toast = ({ msg, type, onClose }) => (
    <div style={{
        position: "fixed", bottom: 24, right: 24, zIndex: 9999,
        padding: "12px 18px", borderRadius: 12,
        background: type === "error" ? "rgba(239,68,68,.15)" : "rgba(34,197,94,.15)",
        border: `1px solid ${type === "error" ? "#ef444460" : "#22c55e60"}`,
        color: type === "error" ? "#ef4444" : C.green,
        display: "flex", alignItems: "center", gap: 10, fontSize: 13, fontWeight: 500,
        boxShadow: "0 8px 32px rgba(0,0,0,.35)", animation: "fadeSlideUp .3s ease", maxWidth: "90vw",
    }}>
        {type === "error" ? <AlertCircle size={15} /> : <CheckCircle size={15} />}
        {msg}
        <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "inherit", opacity: .6, marginLeft: 4 }}><X size={13} /></button>
    </div>
);

const EmptyState = ({ icon: Icon, title, sub, action }) => (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "60px 24px", textAlign: "center", gap: 12 }}>
        <div style={{ width: 56, height: 56, borderRadius: 16, background: C.inputBg, border: `1px solid ${C.border}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Icon size={24} color={C.sub} />
        </div>
        <div style={{ fontSize: 16, fontWeight: 700, color: C.text }}>{title}</div>
        <div style={{ fontSize: 13, color: C.sub, maxWidth: 280, lineHeight: 1.6 }}>{sub}</div>
        {action}
    </div>
);

const ModelDropdown = ({ activeModel, onChange }) => {
    const [open, setOpen] = useState(false);
    const ref = useRef();
    const m = MODELS[activeModel] || MODELS.mpnet;

    // Close dropdown when clicking outside
    useEffect(() => {
        const handler = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
        document.addEventListener("mousedown", handler);
        return () => document.removeEventListener("mousedown", handler);
    }, []);

    return (
        <div ref={ref} style={{ position: "relative" }}>
            {/* Trigger button */}
            <button onClick={() => setOpen(o => !o)} style={{
                display: "flex", alignItems: "center", gap: 7,
                padding: "7px 12px", borderRadius: 10,
                background: `${m.color}14`, border: `1px solid ${m.color}33`,
                cursor: "pointer", fontFamily: "inherit", transition: "background .15s",
            }}>
                {/* Live indicator dot */}
                <div style={{ width: 7, height: 7, borderRadius: "50%", background: m.color, boxShadow: `0 0 6px ${m.color}` }} />
                <span style={{ fontSize: 11, fontWeight: 700, color: m.color }}>{m.name}</span>
                {/* Chevron — vertically centred with translateY so it aligns with text */}
                <ChevronDown size={11} color={m.color} style={{
                    transition: "transform .18s",
                    transform: open ? "rotate(180deg)" : "rotate(0deg)",
                    display: "block",
                    marginTop: 1,   /* fine-tune arrow alignment with text */
                }} />
            </button>

            {/* Dropdown panel */}
            {open && (
                <div style={{
                    position: "absolute", top: "calc(100% + 8px)", right: 0, width: "min(310px, 92vw)",
                    borderRadius: 14, overflow: "hidden",
                    background: C.drawerBg, border: `1px solid ${C.border}`,
                    boxShadow: "0 16px 48px rgba(0,0,0,.35)", zIndex: 200,
                    animation: "fadeSlideUp .15s ease",
                }}>
                    <div style={{ padding: "10px 14px 7px", fontSize: 10, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".08em" }}>
                        Choose Embedding Model
                    </div>
                    {Object.values(MODELS).map(opt => {
                        const isActive = activeModel === opt.key;
                        return (
                            <button key={opt.key} onClick={() => { onChange(opt.key); setOpen(false); }} style={{
                                width: "100%", display: "flex", alignItems: "flex-start", gap: 11,
                                padding: "12px 14px", border: "none", borderTop: `1px solid ${C.border}`,
                                background: isActive ? `${opt.color}0c` : "transparent",
                                cursor: "pointer", textAlign: "left", fontFamily: "inherit",
                                transition: "background .15s",
                            }}
                                onMouseEnter={e => { if (!isActive) e.currentTarget.style.background = `${opt.color}08`; }}
                                onMouseLeave={e => { if (!isActive) e.currentTarget.style.background = "transparent"; }}
                            >
                                <div style={{ width: 32, height: 32, borderRadius: 8, background: `${opt.color}18`, border: `1px solid ${opt.color}28`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 1 }}>
                                    <Activity size={13} color={opt.color} />
                                </div>
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2, flexWrap: "wrap" }}>
                                        <span style={{ fontSize: 13, fontWeight: 700, color: C.text }}>{opt.name}</span>
                                        <span style={{ padding: "1px 7px", borderRadius: 10, fontSize: 9, fontWeight: 700, background: `${opt.color}18`, color: opt.color }}>{opt.badge}</span>
                                        {isActive && <span style={{ padding: "1px 7px", borderRadius: 10, fontSize: 9, fontWeight: 700, background: `${opt.color}22`, color: opt.color }}>● Active</span>}
                                    </div>
                                    <div style={{ fontSize: 9, fontFamily: "monospace", color: C.sub, marginBottom: 3, wordBreak: "break-all" }}>{opt.short}</div>
                                    <div style={{ fontSize: 11, color: C.sub, lineHeight: 1.45 }}>{opt.desc}</div>
                                </div>
                            </button>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

// GPU-composited animation via translate3d + will-change for smooth open
const ProfileModal = ({ profile, onSave, onClose }) => {
    const [form, setForm] = useState({ ...profile });
    return (
        <>
            {/* Backdrop — GPU-accelerated opacity fade */}
            <div onClick={onClose} style={{
                position: "fixed", inset: 0,
                background: "rgba(0,0,0,.6)",
                backdropFilter: "blur(6px)",
                WebkitBackdropFilter: "blur(6px)",
                zIndex: 400,
                animation: "fadeIn .2s ease",
                willChange: "opacity",
            }} />

            {/* Modal panel — translate3d triggers GPU layer */}
            <div style={{
                position: "fixed",
                top: "50%", left: "50%",
                transform: "translate3d(-50%, -50%, 0)",
                width: "min(400px, 92vw)",
                background: C.drawerBg, borderRadius: 20, padding: 26,
                zIndex: 410, border: `1px solid ${C.border}`,
                boxShadow: "0 24px 64px rgba(0,0,0,.5)",
                animation: "modalIn .22s cubic-bezier(.34,1.2,.64,1)",
                willChange: "transform, opacity",
            }}>
                {/* Header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 22 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
                        <div style={{ width: 42, height: 42, borderRadius: 11, background: `linear-gradient(135deg,${C.blue},#a78bfa)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15, fontWeight: 800, color: "#fff" }}>
                            {(form.name || "U").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
                        </div>
                        <div>
                            <div style={{ fontSize: 15, fontWeight: 700, color: C.text }}>Edit Profile</div>
                            <div style={{ fontSize: 11, color: C.sub }}>Update your details</div>
                        </div>
                    </div>
                    <button onClick={onClose} style={{ width: 28, height: 28, borderRadius: 7, border: `1px solid ${C.border}`, background: C.inputBg, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: C.sub }}>
                        <X size={13} />
                    </button>
                </div>

                {/* Fields */}
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {[
                        { label: "Full Name", key: "name", placeholder: "Your name" },
                        { label: "Email", key: "email", placeholder: "you@email.com" },
                        { label: "Role", key: "role", placeholder: "e.g. Recruiter, HR Manager" },
                        { label: "Organisation", key: "org", placeholder: "Company / College" },
                    ].map(({ label, key, placeholder }) => (
                        <div key={key}>
                            <label style={{ fontSize: 10, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".06em", display: "block", marginBottom: 5 }}>{label}</label>
                            <input value={form[key] || ""} onChange={e => setForm(p => ({ ...p, [key]: e.target.value }))} placeholder={placeholder}
                                style={{ width: "100%", background: C.inputBg, border: `1px solid ${C.border}`, borderRadius: 9, color: C.text, fontSize: 13, padding: "9px 12px", outline: "none", fontFamily: "inherit", boxSizing: "border-box" }} />
                        </div>
                    ))}
                </div>

                {/* Actions */}
                <div style={{ display: "flex", gap: 9, marginTop: 20 }}>
                    <button onClick={onClose} style={{ flex: 1, padding: "10px", borderRadius: 10, border: `1px solid ${C.border}`, background: C.inputBg, color: C.sub, cursor: "pointer", fontSize: 13, fontWeight: 600, fontFamily: "inherit" }}>Cancel</button>
                    <button onClick={() => { onSave(form); onClose(); }} style={{ flex: 2, padding: "10px", borderRadius: 10, border: "none", background: `linear-gradient(135deg,${C.blue},#6366f1)`, color: "#fff", cursor: "pointer", fontSize: 13, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", gap: 7, fontFamily: "inherit" }}>
                        <Save size={13} /> Save Changes
                    </button>
                </div>
            </div>
        </>
    );
};

const Drawer = ({ candidate: c, onClose, isMobile }) => {
    if (!c) return null;
    const matched = c.matched_skills || [];
    // Use server's missing_skills list; if absent fall back to filtering from skills array
    const missing = Array.isArray(c.missing_skills)
        ? c.missing_skills
        : (c.skills || []).filter(s => !matched.includes(s));
    const skScore = Math.round((c.skillScore || 0) * 100);
    const seScore = Math.round((c.semanticScore || 0) * 100);
    const matchedSet = new Set(matched.map(x => x.toLowerCase()));
    // All required skills — no truncation; binary: matched = 100, missing = 0
    const radarSkills = [...new Set([...matched, ...missing])];
    const totalRequired = radarSkills.length;
    const matchedCount = matched.length;
    const radarData = radarSkills.map(s => ({
        skill: s.length > 16 ? s.slice(0, 14) + "…" : s,
        fullName: s,           // untruncated — used by tooltip
        score: matchedSet.has(s.toLowerCase()) ? 100 : 0,
        fullMark: 100,
    }));

    return (
        <>
            <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,.6)", backdropFilter: "blur(4px)", zIndex: 200 }} />
            <div style={{
                position: "fixed", right: 0, top: 0, bottom: 0,
                width: isMobile ? "100vw" : 420,
                background: C.drawerBg, borderLeft: `1px solid ${C.border}`,
                zIndex: 210, overflowY: "auto", padding: 22,
                animation: "slideInRight .28s cubic-bezier(.4,0,.2,1)",
                display: "flex", flexDirection: "column", gap: 16,
            }}>
                {/* Header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <div style={{ width: 36, height: 36, borderRadius: 9, background: `linear-gradient(135deg,${C.blue},#a78bfa)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 800, color: "#fff", flexShrink: 0 }}>
                            {(c.name || "?").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
                        </div>
                        <div>
                            <div style={{ fontSize: 16, fontWeight: 700, color: C.text }}>{c.name}</div>
                            <div style={{ fontSize: 11, color: C.sub }}>{c.filename}</div>
                        </div>
                    </div>
                    <button onClick={onClose} style={{ width: 28, height: 28, borderRadius: 7, border: `1px solid ${C.border}`, background: C.inputBg, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: C.sub }}>
                        <X size={13} />
                    </button>
                </div>

                {/* Scores */}
                <div style={card()}>
                    <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 14 }}>Score Breakdown</div>
                    <div style={{ display: "flex", justifyContent: "space-around" }}>
                        {[
                            { label: "Skill", value: Math.round((c.skillScore || 0) * 100), color: C.blue },
                            { label: "Semantic", value: Math.round((c.semanticScore || 0) * 100), color: C.teal },
                            { label: "Final", value: Math.round((c.finalScore || 0) * 100), color: C.amber },
                        ].map(({ label, value, color }) => (
                            <div key={label} style={{ textAlign: "center" }}>
                                <CircularRing value={value} color={color} />
                                <div style={{ fontSize: 10, color: C.sub, marginTop: 7 }}>{label}</div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Info rows */}
                <div style={card({ padding: "4px 16px" })}>
                    {(() => {
                        const rows = [
                            { icon: Mail, label: "Email", value: c.email || "Not Found", href: c.email ? `mailto:${c.email}` : null },
                            { icon: Clock, label: "Experience", value: !c.experience ? "Fresher" : `${c.experience} yrs`, href: null },
                            { icon: Phone, label: "Phone", value: c.phone || "Not found", href: c.phone ? `tel:${c.phone.replace(/\s/g, "")}` : null },
                        ];
                        return rows.map(({ icon: Icon, label, value, href }, idx) => (
                            <div key={label} style={{ display: "flex", gap: 10, padding: "9px 0", borderBottom: idx < rows.length - 1 ? `1px solid ${C.border}` : "none", alignItems: "center" }}>
                                <Icon size={13} color={C.sub} style={{ flexShrink: 0 }} />
                                <div style={{ flex: 1 }}>
                                    <div style={{ fontSize: 10, color: C.sub }}>{label}</div>
                                    {href
                                        ? <a href={href} style={{ fontSize: 13, color: C.blue, marginTop: 1, display: "block", textDecoration: "none", fontWeight: 500 }}
                                            onMouseEnter={e => e.currentTarget.style.textDecoration = "underline"}
                                            onMouseLeave={e => e.currentTarget.style.textDecoration = "none"}>{value}</a>
                                        : <div style={{ fontSize: 13, color: C.cardText, marginTop: 1 }}>{value}</div>
                                    }
                                </div>
                            </div>
                        ));
                    })()}
                    <div style={{ padding: "9px 0", borderTop: `1px solid ${C.border}` }}>
                        <div style={{ fontSize: 10, color: C.sub, marginBottom: 5 }}>Status</div>
                        <Badge status={c.eligible ? "Shortlisted" : "Rejected"} />
                        {!c.eligible && c.rejection_reason && <div style={{ fontSize: 11, color: "#ef4444", marginTop: 5 }}>{c.rejection_reason}</div>}
                    </div>
                </div>

                {/* Links — LinkedIn / GitHub / Portfolio */}
                {(c.links?.linkedin || c.links?.github || c.links?.portfolio) && (
                    <div style={card({ padding: "4px 16px 12px" })}>
                        <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", padding: "8px 0 4px" }}>Links</div>
                        {(() => {
                            const items = [
                                { icon: ExternalLink, label: "LinkedIn", value: c.links?.linkedin, color: "#0A66C2" },
                                { icon: ExternalLink, label: "GitHub", value: c.links?.github, color: C.text },
                                { icon: Globe, label: "Portfolio", value: c.links?.portfolio, color: C.teal },
                            ].filter(l => l.value);
                            return items.map(({ icon: Icon, label, value, color }, idx) => (
                                <div key={label} style={{ display: "flex", gap: 10, padding: "8px 0", borderBottom: idx < items.length - 1 ? `1px solid ${C.border}` : "none", alignItems: "center" }}>
                                    <Icon size={13} color={color} style={{ flexShrink: 0 }} />
                                    <div style={{ flex: 1, minWidth: 0 }}>
                                        <div style={{ fontSize: 10, color: C.sub }}>{label}</div>
                                        <a href={value} target="_blank" rel="noopener noreferrer"
                                            style={{ fontSize: 12, color: C.blue, display: "block", textDecoration: "none", fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                                            onMouseEnter={e => e.currentTarget.style.textDecoration = "underline"}
                                            onMouseLeave={e => e.currentTarget.style.textDecoration = "none"}>
                                            {value.replace(/^https?:\/\//, "")}
                                        </a>
                                    </div>
                                </div>
                            ));
                        })()}
                    </div>
                )}

                {/* Radar chart — same quality as analytics top-3 */}
                {radarData.length >= 3 && (
                    <div style={card({ padding: 16 })}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
                            <div style={{ fontSize: 11, fontWeight: 800, color: C.text }}>Skill Radar</div>
                            <div style={{ fontSize: 9, color: C.muted }}>Skill {skScore}% · Semantic {seScore}%</div>
                        </div>
                        <div style={{ fontSize: 10, color: C.muted, marginBottom: 8 }}>
                            Filled area = skill match coverage · Hover points to see scores
                        </div>
                        <ResponsiveContainer width="100%" height={Math.max(260, radarSkills.length * 22)}>
                            <RadarChart data={radarData} outerRadius="68%" margin={{ top: 12, right: 22, bottom: 12, left: 22 }}>
                                <PolarGrid gridType="circle" stroke={C.border} />
                                <PolarAngleAxis dataKey="skill" tick={{ fill: C.text, fontSize: 9, fontWeight: 600 }} />
                                <Tooltip
                                    content={({ payload }) => {
                                        if (!payload?.length) return null;
                                        const { fullName, score } = payload[0].payload;
                                        const isMatched = score === 100;
                                        const coveragePct = Math.round(100 / totalRequired);
                                        return (
                                            <div style={{ background: "rgba(18,18,22,0.97)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", border: `1px solid ${isMatched ? C.teal : "#ef4444"}`, borderRadius: 10, padding: "9px 13px", fontSize: 11, color: "#f1f1f3", minWidth: 170, boxShadow: "0 4px 16px rgba(0,0,0,.45)" }}>
                                                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 5 }}>
                                                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: isMatched ? C.teal : "#ef4444", flexShrink: 0 }} />
                                                    <span style={{ fontWeight: 800, color: isMatched ? C.teal : "#ef4444", fontSize: 11 }}>{isMatched ? "Matched" : "Missing"}</span>
                                                </div>
                                                <div style={{ fontWeight: 700, color: "#f1f1f3", marginBottom: 6, lineHeight: 1.3 }}>{fullName}</div>
                                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, padding: "5px 0", borderTop: `1px solid rgba(255,255,255,0.10)` }}>
                                                    <span style={{ color: "#9ca3af", fontSize: 10 }}>This skill</span>
                                                    <span style={{ fontWeight: 700, color: "#f1f1f3", fontVariantNumeric: "tabular-nums" }}>1 of {totalRequired} required</span>
                                                </div>
                                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
                                                    <span style={{ color: "#9ca3af", fontSize: 10 }}>Overall skills</span>
                                                    <span style={{ fontWeight: 700, color: isMatched ? C.teal : "#ef4444", fontVariantNumeric: "tabular-nums" }}>{matchedCount}/{totalRequired} matched</span>
                                                </div>
                                            </div>
                                        );
                                    }}
                                />
                                <Radar dataKey="score" stroke={C.blue} strokeWidth={2} fill={C.blue} fillOpacity={0.18}
                                    isAnimationActive={false}
                                    dot={{ fill: C.blue, r: 4, strokeWidth: 1.5, stroke: "rgba(255,255,255,0.2)" }}
                                    activeDot={{ r: 6, fill: C.blue, stroke: "#fff", strokeWidth: 2 }} />
                            </RadarChart>
                        </ResponsiveContainer>
                        {/* Legend */}
                        <div style={{ display: "flex", gap: 20, justifyContent: "center", marginTop: 6, paddingTop: 6, borderTop: `1px solid ${C.border}` }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: C.text }}>
                                {/* Solid filled square = matched skill */}
                                <div style={{ width: 10, height: 10, borderRadius: 2, background: C.blue, flexShrink: 0 }} />
                                <span>Matched skill</span>
                            </div>
                            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: C.muted }}>
                                <div style={{ width: 10, height: 10, borderRadius: 2, border: `1.5px dashed ${C.blue}80`, background: "transparent", flexShrink: 0 }} />
                                <span>Missing skill</span>
                            </div>
                        </div>
                    </div>
                )}

                {/* Skills */}
                <div style={card()}>
                    <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 11 }}>Skills</div>
                    {matched.length > 0 && (
                        <div style={{ marginBottom: 10 }}>
                            <div style={{ fontSize: 11, color: C.teal, fontWeight: 600, marginBottom: 6 }}>✓ Matched ({matched.length})</div>
                            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>{matched.map(s => <SkillChip key={s} label={s} matched />)}</div>
                        </div>
                    )}
                    {missing.length > 0 && (
                        <div>
                            <div style={{ fontSize: 11, color: "#ef4444", fontWeight: 600, marginBottom: 6 }}>✗ Missing ({missing.length})</div>
                            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>{missing.map(s => <SkillChip key={s} label={s} matched={false} />)}</div>
                        </div>
                    )}
                </div>
            </div>
        </>
    );
};


// ── JD quality validator — runs purely client-side, no backend call ─────────
// Returns { score 0-4, label, color, reason } for the job description text.
function validateJD(text) {
    const t = (text || "").trim();
    if (!t) return { score: 0, label: "Empty", color: "#ef4444", reason: "Enter a job description to continue." };

    const words = t.split(/\s+/).filter(w => w.length > 1);
    const wordCount = words.length;

    // Gibberish check: vowel ratio in alphabetic characters
    const alpha = t.toLowerCase().replace(/[^a-z]/g, "");
    const vowels = (alpha.match(/[aeiou]/g) || []).length;
    const vowelRatio = alpha.length > 0 ? vowels / alpha.length : 0;

    // Unique char ratio (random strings score very high)
    const uniqueRatio = alpha.length > 0 ? new Set(alpha).size / alpha.length : 1;

    // Common English word presence
    const COMMON = new Set(["the","a","an","and","or","to","of","in","for","with",
        "is","are","we","you","this","will","be","as","work","role","team","skills",
        "experience","knowledge","ability","candidate","position","need","looking",
        "required","develop","build","manage","design","degree","years","good"]);
    const hasEnglish = words.slice(0, 40).some(w => COMMON.has(w.toLowerCase().replace(/[^a-z]/g, "")));

    if (wordCount < 6 || vowelRatio < 0.08 || (uniqueRatio > 0.65 && t.length < 120) || !hasEnglish) {
        return { score: 0, label: "Invalid", color: "#ef4444",
            reason: wordCount < 6
                ? `Too short — only ${wordCount} word(s). Write at least a sentence describing the role.`
                : "Appears to be random text. Please describe the role in plain English." };
    }
    if (wordCount < 20) return { score: 1, label: "Weak", color: "#f97316", reason: "Very brief — results may be inaccurate. Add responsibilities and requirements." };
    if (wordCount < 50) return { score: 2, label: "Fair", color: C.amber, reason: "Moderate detail. Adding specific skills, tools, and responsibilities improves accuracy." };
    if (wordCount < 100) return { score: 3, label: "Good", color: C.teal, reason: "Good detail. More specific technical terms will further improve matching." };
    return { score: 4, label: "Excellent", color: C.green, reason: "Detailed job description — optimal for accurate screening." };
}

// Validate a single skill name — returns null if valid, error string if invalid
function validateSkillInput(raw) {
    const s = raw.trim();
    if (s.length < 2) return "Skill name must be at least 2 characters.";
    if (s.length > 50) return "Skill name too long (max 50 characters).";
    if (!/[a-zA-Z]/.test(s)) return "Skill must contain at least one letter.";
    const alpha = s.replace(/[^a-z]/gi, "").toLowerCase();
    if (alpha.length >= 6) {
        const vowels = (alpha.match(/[aeiou]/g) || []).length;
        if (vowels / alpha.length < 0.05) return `"${s}" doesn't look like a real skill name.`;
    }
    return null;
}

const UploadView = ({ onStartScreening, activeModel, onModelChange, isMobile, backendOnline }) => {
    const [fileItems, setFileItems] = useState([]);
    const [isDragging, setIsDragging] = useState(false);
    const [toast, setToast] = useState(null);
    const [jd, setJd] = useState("");
    const [skills, setSkills] = useState([]);
    const [newSkill, setNewSkill] = useState("");
    const [minExp, setMinExp] = useState(0);
    const [skillWeight, setSkillWeight] = useState(55);
    const [configLoaded, setConfigLoaded] = useState(false);
    const fileRef = useRef();
    const jdRef = useRef();

    const MAX_FILES = 100;
    const MAX_SIZE_MB = 10;

    // Load saved config from backend once when the upload form mounts
    useEffect(() => {
        fetch(`${BASE}/api/config`)
            .then(r => r.json())
            .then(data => {
                if (!data) return;
                let loaded = false;
                if (data.job_description) { setJd(data.job_description); loaded = true; }
                if (data.required_skills?.length) { setSkills(data.required_skills); loaded = true; }
                if (data.scoring?.min_experience_years !== undefined)
                    setMinExp(data.scoring.min_experience_years);
                if (data.scoring?.skill_weight !== undefined)
                    setSkillWeight(Math.round(data.scoring.skill_weight * 100));
                if (loaded) setConfigLoaded(true);
            })
            .catch(() => { /* silently ignore if backend is offline */ });
    }, []); // BASE is a module-level constant — intentional empty deps

    const showToast = (msg, type = "success") => {
        setToast({ msg, type });
        setTimeout(() => setToast(null), type === "error" ? 7000 : 3000);
    };

    const addFiles = useCallback((incoming) => {
        const arr = Array.from(incoming);
        const valid = arr.filter(f => f.type === "application/pdf");
        const bad = arr.filter(f => f.type !== "application/pdf");
        const tooBig = valid.filter(f => f.size > MAX_SIZE_MB * 1024 * 1024);
        const okSize = valid.filter(f => f.size <= MAX_SIZE_MB * 1024 * 1024);

        if (bad.length) showToast(`${bad.length} non-PDF file(s) were skipped`, "error");
        if (tooBig.length) showToast(`${tooBig.length} file(s) exceed the ${MAX_SIZE_MB} MB size limit`, "error");
        if (!okSize.length) return;

        // All toast logic computed outside setFileItems — side-effects inside setState are not safe
        setFileItems(prev => {
            if (prev.length >= MAX_FILES) {
                showToast(`You can upload a maximum of ${MAX_FILES} resumes per screening session`, "error");
                return prev;
            }

            // Deduplicate by name+size so two different files with the same name are both accepted
            const existing = new Set(prev.map(f => `${f.name}::${f.rawSize}`));
            const candidates = okSize.filter(f => !existing.has(`${f.name}::${f.size}`));

            // Trim to remaining capacity — never exceed MAX_FILES
            const remaining = MAX_FILES - prev.length;
            const fresh = candidates.slice(0, remaining).map(f => ({
                id: `${f.name}-${f.size}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
                name: f.name,
                rawSize: f.size,
                raw: f,
                size: f.size > 1048576 ? (f.size / 1048576).toFixed(1) + " MB" : (f.size / 1024).toFixed(0) + " KB",
            }));

            const dupCount = okSize.length - candidates.length;
            const cappedCount = candidates.length - fresh.length;
            if (dupCount > 0) showToast(`${dupCount} duplicate file(s) were skipped`, "error");
            if (cappedCount > 0) showToast(`${cappedCount} file(s) not added — session is full (${MAX_FILES} max)`, "error");
            if (fresh.length > 0) {
                showToast(`${fresh.length} PDF${fresh.length !== 1 ? "s" : ""} added successfully`, "success");
                // Only auto-scroll to JD for small batches; large batches should stay
                // on the file list so the user can confirm what was uploaded.
                const totalAfter = prev.length + fresh.length;
                if (totalAfter <= 5) {
                    setTimeout(() => jdRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 300);
                }
            }
            return [...prev, ...fresh];
        });
    }, []);

    // ── Folder drag-and-drop helpers ─────────────────────────────────────
    // Recursively collect all .pdf files from a FileSystemEntry (file or directory).
    // readEntries() only returns ≪100 entries at a time — we loop until empty.
    const collectPdfsFromEntry = useCallback(async (entry) => {
        if (!entry) return [];
        if (entry.isFile) {
            return new Promise((resolve) => {
                entry.file(
                    (f) => {
                        const isPdf = f.name.toLowerCase().endsWith(".pdf") ||
                                      f.type === "application/pdf";
                        resolve(isPdf ? [f] : []);
                    },
                    () => resolve([]),  // getFile() error — skip silently
                );
            });
        }
        if (entry.isDirectory) {
            const reader = entry.createReader();
            const allEntries = [];
            // readEntries batches at ≪100 — keep calling until we get an empty batch
            await new Promise((resolve) => {
                const readBatch = () =>
                    reader.readEntries(
                        (batch) => { if (!batch.length) return resolve(); allEntries.push(...batch); readBatch(); },
                        () => resolve(),  // readEntries error — stop gracefully
                    );
                readBatch();
            });
            const nested = await Promise.all(allEntries.map(collectPdfsFromEntry));
            return nested.flat();
        }
        return [];
    }, []);

    const folderRef = useRef();

    const onDrop = useCallback(async (e) => {
        e.preventDefault();
        setIsDragging(false);
        const items = Array.from(e.dataTransfer.items || []);
        // Use the FileSystemEntry API when available (Chrome, Firefox, Safari, Edge ≥ 2020)
        if (items.length && typeof items[0].webkitGetAsEntry === "function") {
            const entries   = items.map((i) => i.webkitGetAsEntry()).filter(Boolean);
            const hasFolder = entries.some((en) => en.isDirectory);
            const allFiles  = (await Promise.all(entries.map(collectPdfsFromEntry))).flat();
            if (hasFolder && allFiles.length === 0) {
                showToast("No PDF files found inside the dropped folder", "error");
            } else {
                if (hasFolder)
                    showToast(`Found ${allFiles.length} PDF${allFiles.length !== 1 ? "s" : ""} in folder — adding…`, "success");
                addFiles(allFiles);
            }
        } else {
            // Fallback for browsers without FileSystemEntry API — flat file list only
            addFiles(e.dataTransfer.files);
        }
    }, [addFiles, collectPdfsFromEntry]);

    const jdQuality = validateJD(jd);
    const canStart = fileItems.length > 0 && jdQuality.score > 0 && backendOnline !== false;
    const activeM = MODELS[activeModel] || MODELS.mpnet;

    // Parse comma-separated input and add unique skills — with validation
    const addSkill = (raw = newSkill) => {
        const parts = raw.split(",").map(s => s.trim()).filter(s => s.length > 0);
        const invalid = [];
        const toAdd = [];
        parts.forEach(p => {
            const err = validateSkillInput(p);
            if (err) invalid.push(err);
            else toAdd.push(p.toLowerCase());
        });
        if (invalid.length) {
            showToast(invalid[0], "error");
        }
        if (toAdd.length) {
            setSkills(prev => { const ex = new Set(prev); return [...prev, ...toAdd.filter(s => !ex.has(s))]; });
        }
        setNewSkill("");
    };
    const handleSkillKeyDown = (e) => { if (e.key === "Enter") { e.preventDefault(); addSkill(); } };

    const handleStart = () => {
        if (!canStart) return;
        if (jdQuality.score === 0) {
            showToast(jdQuality.reason, "error");
            return;
        }
        onStartScreening({ jd, skills, minExp, skillWeight, fileCount: fileItems.length, rawFiles: fileItems.map(f => f.raw), model: activeModel });
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
            <div>
                <h2 style={{ fontSize: 21, fontWeight: 800, color: C.text, margin: 0 }}>Upload & Configure</h2>
                <p style={{ color: C.sub, marginTop: 5, fontSize: 13 }}>Add PDF resumes, set your job requirements, and let the ML pipeline rank your candidates.</p>
            </div>

            {/* Config loaded notice — dismissible */}
            {configLoaded && (
                <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", borderRadius: 10, background: `${C.blue}08`, border: `1px solid ${C.blue}22`, fontSize: 12 }}>
                    <Info size={13} color={C.blue} style={{ flexShrink: 0 }} />
                    <span style={{ flex: 1, color: C.sub }}>
                        <strong style={{ color: C.blue }}>Pre-filled from saved Job Config.</strong> The values below are editable — what you see here is exactly what gets sent to the pipeline. Changes here do <em>not</em> affect the saved config.
                    </span>
                    <button onClick={() => setConfigLoaded(false)} style={{ background: "none", border: "none", cursor: "pointer", color: C.muted, padding: 2, display: "flex" }}><X size={12} /></button>
                </div>
            )}

            {/* Backend offline banner */}
            {backendOnline === false && (
                <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 16px", borderRadius: 12, background: "rgba(239,68,68,.08)", border: "1px solid rgba(239,68,68,.25)", fontSize: 13, color: "#ef4444" }}>
                    <AlertCircle size={15} style={{ flexShrink: 0 }} />
                    <span>The Flask backend is not running. Start it with <code style={{ fontFamily: "monospace", background: "rgba(239,68,68,.12)", padding: "1px 6px", borderRadius: 4 }}>python app.py</code> before running a screening.</span>
                </div>
            )}

            {/* Drop zone */}
            <div onClick={() => fileRef.current.click()}
                onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
                onMouseEnter={e => { if (!isDragging) e.currentTarget.style.borderColor = C.blue; }}
                onMouseLeave={e => { if (!isDragging) e.currentTarget.style.borderColor = C.border; }}
                style={{ border: `2px dashed ${isDragging ? C.blue : C.border}`, borderRadius: 18, padding: "36px 28px", cursor: "pointer", textAlign: "center", background: isDragging ? `${C.blue}08` : C.inputBg, display: "flex", flexDirection: "column", alignItems: "center", gap: 10, transition: "all .2s" }}>
                <div style={{ width: 54, height: 54, borderRadius: 14, background: isDragging ? `${C.blue}20` : C.surface, border: `1px solid ${isDragging ? C.blue : C.border}`, display: "flex", alignItems: "center", justifyContent: "center", transition: "all .2s" }}>
                    <Upload size={22} color={isDragging ? C.blue : C.sub} />
                </div>
                <div>
                    <div style={{ fontSize: 15, fontWeight: 600, color: isDragging ? C.blue : C.cardText }}>
                        {isDragging ? "Release to upload" : "Drag & drop resumes or an entire folder here"}
                    </div>
                    <div style={{ fontSize: 12, color: C.sub, marginTop: 3 }}>
                        Folders are scanned recursively for PDFs · use buttons below to browse
                    </div>
                </div>
                {/* Limits notice */}
                <div style={{ display: "flex", gap: 12, marginTop: 2 }}>
                    {[`Max ${MAX_FILES} files`, `Max ${MAX_SIZE_MB} MB each`, "PDF only"].map(t => (
                        <span key={t} style={{ fontSize: 10, color: C.muted, background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6, padding: "2px 8px" }}>{t}</span>
                    ))}
                </div>
                {/* Hidden inputs: one for multi-file, one for whole-folder browse */}
                <input ref={fileRef} type="file" accept=".pdf" multiple hidden onChange={e => addFiles(e.target.files)} />
                <input ref={folderRef} type="file" accept=".pdf" /* @ts-ignore */
                    webkitdirectory="" mozdirectory="" directory="" multiple hidden
                    onChange={e => addFiles(e.target.files)} />
                <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
                    <button
                        type="button"
                        onClick={e => { e.stopPropagation(); fileRef.current.click(); }}
                        style={{
                            padding: "5px 14px", borderRadius: 8, fontSize: 11, fontWeight: 600,
                            background: `${C.blue}14`, border: `1px solid ${C.blue}33`,
                            color: C.blue, cursor: "pointer", fontFamily: "inherit",
                        }}
                    >
                        Browse Files
                    </button>
                    <button
                        type="button"
                        onClick={e => { e.stopPropagation(); folderRef.current.click(); }}
                        style={{
                            padding: "5px 14px", borderRadius: 8, fontSize: 11, fontWeight: 600,
                            background: `${C.teal}14`, border: `1px solid ${C.teal}33`,
                            color: C.teal, cursor: "pointer", fontFamily: "inherit",
                        }}
                    >
                        Browse Folder
                    </button>
                </div>
            </div>

            {/* Uploaded file list — sticky header with scrollable body */}
            {fileItems.length > 0 && (
                <div style={card({ padding: 0, overflow: "hidden" })}>
                    {/* Header always visible — not clipped by the scrollable list below */}
                    <div style={{
                        padding: "12px 16px",
                        borderBottom: `1px solid ${C.border}`,
                        display: "flex", justifyContent: "space-between", alignItems: "center",
                        background: C.surface,
                        position: "sticky", top: 0, zIndex: 2,
                    }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <div style={{
                                width: 24, height: 24, borderRadius: 6,
                                background: `${C.green}18`, border: `1px solid ${C.green}30`,
                                display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                            }}>
                                <CheckCircle size={12} color={C.green} />
                            </div>
                            <span style={{ fontSize: 13, fontWeight: 700, color: C.text }}>
                                {fileItems.length} {fileItems.length === 1 ? "File" : "Files"} Uploaded
                            </span>
                            <span style={{
                                fontSize: 10, fontWeight: 600,
                                padding: "2px 8px", borderRadius: 20,
                                background: `${C.green}14`, color: C.green,
                                border: `1px solid ${C.green}25`,
                            }}>Ready</span>
                        </div>
                        <button
                            onClick={() => setFileItems([])}
                            style={{
                                background: "rgba(239,68,68,.08)", border: "1px solid rgba(239,68,68,.22)",
                                borderRadius: 7, cursor: "pointer", fontSize: 11, fontWeight: 600,
                                color: "#ef4444", fontFamily: "inherit", padding: "3px 10px",
                                display: "flex", alignItems: "center", gap: 4,
                            }}
                        >
                            <X size={10} /> Clear All
                        </button>
                    </div>

                    {/* Scrollable file list — capped at 320 px so header stays visible */}
                    <div style={{
                        maxHeight: 320,
                        overflowY: "auto",
                        overflowX: "hidden",
                        scrollbarWidth: "thin",
                        scrollbarColor: `${C.scrollThumb} transparent`,
                    }}>
                        {fileItems.map((f, idx) => (
                            <div key={f.id}
                                style={{
                                    display: "flex", alignItems: "center", gap: 10,
                                    padding: "8px 16px",
                                    borderBottom: `1px solid ${C.border}33`,
                                    transition: "background .12s",
                                }}
                                onMouseEnter={e => e.currentTarget.style.background = C.tableFocus}
                                onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                            >
                                <span style={{
                                    fontSize: 10, color: C.muted, fontVariantNumeric: "tabular-nums",
                                    minWidth: 22, textAlign: "right", flexShrink: 0,
                                }}>{idx + 1}</span>
                                <div style={{
                                    width: 28, height: 28, borderRadius: 7,
                                    background: "rgba(239,68,68,.1)", border: "1px solid rgba(239,68,68,.18)",
                                    display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                                }}>
                                    <FileText size={12} color="#ef4444" />
                                </div>
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{
                                        fontSize: 12, color: C.cardText, fontWeight: 500,
                                        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                                    }}>{f.name}</div>
                                    <div style={{ fontSize: 10, color: C.muted }}>{f.size}</div>
                                </div>
                                <CheckCircle size={11} color={C.green} style={{ flexShrink: 0 }} />
                                <button
                                    onClick={() => setFileItems(p => p.filter(x => x.id !== f.id))}
                                    title="Remove file"
                                    style={{
                                        background: "none", border: "none", cursor: "pointer",
                                        color: C.muted, padding: 2, borderRadius: 4, flexShrink: 0,
                                        display: "flex", alignItems: "center",
                                        transition: "color .12s",
                                    }}
                                    onMouseEnter={e => e.currentTarget.style.color = "#ef4444"}
                                    onMouseLeave={e => e.currentTarget.style.color = C.muted}
                                >
                                    <X size={11} />
                                </button>
                            </div>
                        ))}
                    </div>

                    {/* Footer summary when list overflows */}
                    {fileItems.length > 8 && (
                        <div style={{
                            padding: "8px 16px",
                            borderTop: `1px solid ${C.border}`,
                            display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                            fontSize: 11, color: C.muted,
                            background: C.surface,
                        }}>
                            <Info size={11} />
                            Scroll to view all {fileItems.length} files · ready to screen
                        </div>
                    )}
                </div>
            )}

            {/* Job requirements — two columns on desktop, same height */}
            <div ref={jdRef} style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr", gap: 18, alignItems: "stretch" }}>
                {/* Left column: JD */}
                <div style={{ ...card(), display: "flex", flexDirection: "column" }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 6, display: "flex", alignItems: "center", gap: 7, justifyContent: "space-between" }}>
                        <span style={{ display: "flex", alignItems: "center", gap: 7 }}><Briefcase size={12} /> Job Description</span>
                        {/* Live JD quality badge */}
                        {jd.trim() && (
                            <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 10px", borderRadius: 20, background: `${jdQuality.color}18`, color: jdQuality.color, border: `1px solid ${jdQuality.color}30`, letterSpacing: ".04em" }}>
                                {jdQuality.label}
                            </span>
                        )}
                    </div>
                    {/* Quality hint */}
                    {jd.trim() && jdQuality.score < 3 && (
                        <div style={{ fontSize: 11, color: jdQuality.color, marginBottom: 8, display: "flex", alignItems: "flex-start", gap: 6, lineHeight: 1.5 }}>
                            <AlertCircle size={12} style={{ flexShrink: 0, marginTop: 1 }} />
                            {jdQuality.reason}
                        </div>
                    )}
                    <label style={{ fontSize: 12, color: C.sub, display: "block", marginBottom: 6 }}>Describe the Role *</label>
                    <textarea value={jd} onChange={e => { setJd(e.target.value); e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; }}
                        placeholder="Describe what the candidate will be doing, required responsibilities, and what you expect from them. Be specific for better results."
                        style={{ flex: 1, width: "100%", background: C.inputBg, border: `1px solid ${C.border}`, borderRadius: 10, color: C.cardText, fontSize: 13, padding: "10px 12px", resize: "none", outline: "none", fontFamily: "inherit", lineHeight: 1.6, boxSizing: "border-box", overflow: "hidden", minHeight: 160 }} />
                </div>

                {/* Right column: Skills + sliders */}
                <div style={card()}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 14, display: "flex", alignItems: "center", gap: 7 }}>
                        <Target size={12} /> Skills & Weights
                    </div>

                    {/* Required Skills header + Clear All */}
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 7 }}>
                        <label style={{ fontSize: 12, color: C.sub }}>Required Skills</label>
                        {skills.length > 0 && (
                            <button onClick={() => setSkills([])} style={{ fontSize: 11, color: "#ef4444", background: "rgba(239,68,68,.08)", border: "1px solid rgba(239,68,68,.2)", borderRadius: 6, padding: "2px 9px", cursor: "pointer", fontFamily: "inherit", fontWeight: 600 }}>Clear All</button>
                        )}
                    </div>
                    {skills.length > 0 && (
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                            {skills.map(s => (
                                <div key={s} style={{ display: "flex", alignItems: "center", gap: 5, padding: "3px 9px", borderRadius: 20, background: `${C.blue}18`, border: `1px solid ${C.blue}30`, color: C.blue, fontSize: 12, fontWeight: 500 }}>
                                    {s}
                                    <button onClick={() => setSkills(p => p.filter(x => x !== s))} style={{ background: "none", border: "none", cursor: "pointer", color: C.sub, padding: 0, display: "flex" }}><X size={9} /></button>
                                </div>
                            ))}
                        </div>
                    )}
                    <div style={{ display: "flex", gap: 7, marginBottom: 18 }}>
                        <input value={newSkill} onChange={e => setNewSkill(e.target.value)}
                            onKeyDown={handleSkillKeyDown}
                            placeholder="e.g. python, sql, machine learning — press Enter to add"
                            style={{ flex: 1, background: C.inputBg, border: `1px solid ${C.border}`, borderRadius: 8, color: C.cardText, fontSize: 13, padding: "8px 11px", outline: "none", fontFamily: "inherit" }} />
                        <button onClick={() => addSkill()} style={{ padding: "8px 13px", borderRadius: 8, background: `${C.blue}18`, border: `1px solid ${C.blue}30`, color: C.blue, cursor: "pointer", fontFamily: "inherit", fontWeight: 700 }}>
                            <Plus size={13} />
                        </button>
                    </div>

                    {/* Sliders */}
                    {[
                        { label: "Min Experience", value: minExp, set: setMinExp, min: 0, max: 10, step: .5, color: C.amber, fmt: v => v === 0 ? "Freshers — 0 yrs" : `${v}+ yrs` },
                        { label: "Skill Weight", value: skillWeight, set: setSkillWeight, min: 0, max: 100, step: 5, color: C.blue, fmt: v => `${v}% skill / ${100 - v}% semantic` },
                    ].map(({ label, value, set, min, max, step, color, fmt }) => (
                        <div key={label} style={{ marginBottom: 14 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                                <span style={{ fontSize: 12, color: C.sub }}>{label}</span>
                                <span style={{ fontSize: 12, color, fontWeight: 700 }}>{fmt(value)}</span>
                            </div>
                            <input type="range" min={min} max={max} step={step} value={value}
                                onChange={e => set(parseFloat(e.target.value))}
                                style={{ width: "100%", accentColor: color, cursor: "pointer" }} />
                        </div>
                    ))}
                </div>
            </div>

            {/* Model info card + inline model switcher + run button */}
            <div style={{
                borderRadius: 16, overflow: "hidden",
                border: `1px solid ${activeM.color}30`,
                background: `${activeM.color}12`,
            }}>
                {/* Model info header */}
                <div style={{ padding: "16px 20px", display: "flex", alignItems: "flex-start", gap: 14, borderBottom: `1px solid ${activeM.color}18` }}>
                    {/* Left: icon — vertically centred */}
                    <div style={{ width: 42, height: 42, borderRadius: 11, background: `${activeM.color}18`, border: `1px solid ${activeM.color}30`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, alignSelf: "center" }}>
                        <Activity size={18} color={activeM.color} />
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3, flexWrap: "wrap" }}>
                            <span style={{ fontSize: 14, fontWeight: 800, color: C.text }}>{activeM.name}</span>
                            <span style={{ padding: "2px 8px", borderRadius: 10, fontSize: 10, fontWeight: 700, background: `${activeM.color}18`, color: activeM.color, border: `1px solid ${activeM.color}30` }}>{activeM.badge}</span>
                        </div>
                        <div style={{ fontSize: 11, fontFamily: "monospace", color: C.sub, marginBottom: 5, wordBreak: "break-all" }}>{activeM.short}</div>
                        <div style={{ fontSize: 13, color: C.cardText, lineHeight: 1.55 }}>{activeM.desc}</div>
                        <div style={{ fontSize: 11, color: C.sub, marginTop: 5, lineHeight: 1.5 }}>{activeM.detail}</div>
                    </div>
                </div>

                {/* Inline model switcher — centered */}
                <div style={{ padding: "10px 20px", borderBottom: `1px solid ${activeM.color}14`, display: "flex", alignItems: "center", justifyContent: "center", gap: 8, flexWrap: "wrap" }}>
                    <span style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".07em", flexShrink: 0 }}>Switch Active Model:</span>
                    {Object.values(MODELS).map(m => (
                        <button key={m.key} onClick={() => onModelChange(m.key)}
                            style={{
                                padding: "4px 12px", borderRadius: 20, fontSize: 11, fontWeight: 600, cursor: "pointer", fontFamily: "inherit", transition: "all .15s",
                                background: activeModel === m.key ? `${m.color}22` : C.inputBg,
                                color: activeModel === m.key ? m.color : C.sub,
                                border: `1px solid ${activeModel === m.key ? `${m.color}50` : C.border}`,
                            }}>
                            {m.name}
                        </button>
                    ))}
                </div>

                {/* Run button or warning */}
                {canStart ? (
                    <button onClick={handleStart} style={{
                        width: "100%", padding: "16px 24px",
                        border: "none",
                        background: `linear-gradient(135deg,${activeM.color},${activeM.color}99)`,
                        color: "#fff", cursor: "pointer", fontSize: 14, fontWeight: 800,
                        display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                        fontFamily: "inherit", position: "relative", overflow: "hidden",
                        boxShadow: `0 4px 20px ${activeM.color}55`,
                        transition: "box-shadow .2s, transform .1s",
                    }}
                        onMouseEnter={e => { e.currentTarget.style.boxShadow = `0 6px 28px ${activeM.color}88`; e.currentTarget.style.transform = "translateY(-1px)"; }}
                        onMouseLeave={e => { e.currentTarget.style.boxShadow = `0 4px 20px ${activeM.color}55`; e.currentTarget.style.transform = "none"; }}
                        onMouseDown={e => e.currentTarget.style.transform = "scale(.98)"}
                        onMouseUp={e => e.currentTarget.style.transform = "translateY(-1px)"}
                    >
                        {/* shimmer */}
                        <span style={{ position: "absolute", inset: 0, background: "linear-gradient(90deg,transparent,rgba(255,255,255,.18),transparent)", animation: "runShimmer 2.2s ease infinite", pointerEvents: "none" }} />
                        <Zap size={16} style={{ animation: "runPulse 1.6s ease infinite" }} />
                        Run Screening with {activeM.name}
                        <ArrowRight size={15} />
                    </button>
                ) : (
                    <div style={{ padding: "13px 20px", display: "flex", alignItems: "center", justifyContent: "center", gap: 9, fontSize: 12, color: C.sub }}>
                        <Info size={13} color={C.sub} style={{ flexShrink: 0 }} />
                        {backendOnline === false
                            ? "Start the Flask backend (python app.py) before running a screening."
                            : fileItems.length === 0
                                ? "Upload at least one resume PDF to continue."
                                : jdQuality.score === 0
                                    ? jdQuality.reason
                                    : "Add a valid job description (at least 6 words) to continue."}
                    </div>
                )}
            </div>

            {toast && <Toast msg={toast.msg} type={toast.type} onClose={() => setToast(null)} />}
        </div>
    );
};

const ProcessingView = ({ config, onDone, onSessionReady }) => {
    const [progress, setProgress] = useState(0);
    const [step, setStep] = useState(0);
    const [status, setStatus] = useState("Connecting to the screening server…");
    const [error, setError] = useState(null);
    const [warnings, setWarnings] = useState([]);
    const [warnCountdown, setWarnCountdown] = useState(null);
    const [pendingResults, setPendingResults] = useState(null);
    // Refs capture config/callback so the mount effect never goes stale
    const configRef = useRef(config);
    const onDoneRef = useRef(onDone);

    const countdownIvRef = useRef(null);

    // Start 10-second countdown the moment pendingResults is available
    useEffect(() => {
        if (pendingResults === null) return;
        // Clear any existing interval (safety guard)
        if (countdownIvRef.current) clearInterval(countdownIvRef.current);
        setWarnCountdown(10);
        countdownIvRef.current = setInterval(() => {
            setWarnCountdown(prev => {
                if (prev === null || prev <= 1) {
                    clearInterval(countdownIvRef.current);
                    return 0;
                }
                return prev - 1;
            });
        }, 1000);
        return () => {
            if (countdownIvRef.current) clearInterval(countdownIvRef.current);
        };
    }, [pendingResults]);

    // Auto-navigate when countdown hits zero
    useEffect(() => {
        if (warnCountdown === 0 && pendingResults !== null) {
            if (countdownIvRef.current) clearInterval(countdownIvRef.current);
            onDoneRef.current(pendingResults);
        }
    }, [warnCountdown, pendingResults]);
    const doneRef = useRef(false);
    const ivRef = useRef(null);

    // Run the ML pipeline exactly once when this component mounts.
    // configRef / onDoneRef are captured at mount so this never re-runs.
    useEffect(() => {
        if (doneRef.current) return;
        const cfg = configRef.current;
        const onDoneStable = onDoneRef.current;

        // Progress bar: fast until 85%, then trickles while waiting for API
        let current = 0;
        ivRef.current = setInterval(() => {
            current += current < 85 ? 2.0 : 0.1;
            const clamped = Math.min(current, 99);
            setProgress(clamped);
            setStep(Math.min(Math.floor(clamped / 17), PIPELINE_STEPS(cfg?.skillWeight || 55, MODELS[cfg?.model || "mpnet"]?.name || "MPNet").length - 1));
        }, 65);

        const run = async () => {
            try {
                // 1. Upload PDFs
                setStatus("Uploading resume files…");
                const fd = new FormData();
                (cfg.rawFiles || []).forEach(f => fd.append("files", f));
                const upRes = await fetch(`${BASE}/api/upload-resumes`, { method: "POST", body: fd });
                if (!upRes.ok) {
                    const e = await upRes.json();
                    throw new Error(e.error || "Upload failed");
                }
                const upData = await upRes.json();
                // Notify App of this session_id immediately — lets it clear the
                // previous session and register this one for tab-close cleanup.
                onSessionReady?.(upData.session_id);
                const savedCount = (upData.saved || []).length;
                const dupCount = (upData.duplicates || []).length;
                const rejCount = (upData.rejected || []).length;
                if (dupCount > 0 || rejCount > 0) {
                    const parts = [];
                    if (savedCount) parts.push(`${savedCount} uploaded`);
                    if (dupCount) parts.push(`${dupCount} duplicate(s) skipped`);
                    if (rejCount) parts.push(`${rejCount} rejected (oversized or non-PDF)`);
                    setStatus(parts.join(" · ") + " — running ML pipeline…");
                }

                // 2. Start the ML pipeline (returns task_id immediately — async)
                setStatus("Starting ML pipeline…");
                const screenRes = await fetch(`${BASE}/api/screen`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        session_id: upData.session_id,
                        job_description: cfg.jd,
                        required_skills: cfg.skills || [],
                        model: cfg.model || "mpnet",
                        config: {
                            min_experience_years: cfg.minExp || 0,
                            skill_weight: (cfg.skillWeight || 55) / 100,
                            semantic_weight: 1 - (cfg.skillWeight || 55) / 100,
                        },
                    }),
                });
                const screenData = await screenRes.json();
                if (!screenRes.ok) throw new Error(screenData.error || "Screening failed");

                // Poll /api/result/<task_id> until the pipeline finishes
                const taskId = screenData.task_id;
                if (!taskId) throw new Error(screenData.error || "No task ID returned from server");

                setStatus("ML pipeline running — processing resumes…");
                let data = null;
                while (true) {
                    await new Promise(r => setTimeout(r, 1200));
                    const pollRes = await fetch(`${BASE}/api/result/${taskId}`);
                    const pollData = await pollRes.json();
                    if (pollData.status === "done") {
                        data = pollData;
                        break;
                    }
                    if (pollData.status === "error" || pollRes.status === 500) {
                        throw new Error(pollData.error || "Screening pipeline failed");
                    }
                    // Still pending/running — keep polling
                    if (pollData.status === "running") setStatus("ML pipeline running — almost done…");
                }

                // Only show warnings for unreadable PDFs (merged duplicates are silent)
                const warns = [];
                if (data.parse_failures?.length) {
                    warns.push(`${data.parse_failures.length} PDF(s) could not be parsed and were skipped: ${data.parse_failures.join(", ")}`);
                }
                clearInterval(ivRef.current);
                doneRef.current = true;
                setProgress(100);
                setStep(PIPELINE_STEPS(cfg?.skillWeight || 55, MODELS[cfg?.model || "mpnet"]?.name || "MPNet").length - 1);
                setStatus("Done — results ready");

                if (warns.length) {
                    setWarnings(warns);
                    setPendingResults(data.results || []);
                } else {
                    setTimeout(() => onDoneStable(data.results || []), 700);
                }

            } catch (err) {
                clearInterval(ivRef.current);
                setError(err.message);
            }
        };

        run();
        return () => clearInterval(ivRef.current);
    }, []); // intentional mount-only: config/onDone captured via refs

    const pct = Math.round(progress);
    const modelInfo = MODELS[config?.model || "mpnet"];

    if (error) {
        return (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "60vh", gap: 14, textAlign: "center", padding: "24px 16px" }}>
                <div style={{ width: 54, height: 54, borderRadius: 15, background: "rgba(239,68,68,.12)", border: "1px solid rgba(239,68,68,.25)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <AlertCircle size={22} color="#ef4444" />
                </div>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.text }}>Screening Failed</div>
                <div style={{ fontSize: 13, color: "#ef4444", maxWidth: 380, lineHeight: 1.6, background: "rgba(239,68,68,.08)", padding: "10px 16px", borderRadius: 10, border: "1px solid rgba(239,68,68,.2)", wordBreak: "break-word" }}>{error}</div>
                <div style={{ fontSize: 12, color: C.sub }}>Make sure Flask is running: <code style={{ color: C.blue }}>python app.py</code></div>
                <button onClick={() => onDone([])} style={{ padding: "9px 20px", borderRadius: 9, border: `1px solid ${C.border}`, background: C.inputBg, color: C.sub, cursor: "pointer", fontSize: 13, fontFamily: "inherit", fontWeight: 600 }}>
                    ← Go Back
                </button>
            </div>
        );
    }

    return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "calc(100vh - 130px)", padding: "24px 14px" }}>
            <div style={{ width: "100%", maxWidth: 520 }}>
                <div style={{ textAlign: "center", marginBottom: 28 }}>
                    <div style={{ fontSize: 20, fontWeight: 800, color: C.text, marginBottom: 5 }}>Running Screening Pipeline</div>
                    <div style={{ fontSize: 13, color: C.sub }}>
                        Analysing <strong style={{ color: C.blue }}>{config?.fileCount || 0}</strong> resume{(config?.fileCount || 0) !== 1 ? "s" : ""} — {status}
                    </div>
                </div>

                {/* Progress indicator — circular ring above a linear bar */}
                <div style={card({ display: "flex", flexDirection: "column", alignItems: "center", gap: 18, marginBottom: 12 })}>
                    <div style={{ position: "relative", width: 112, height: 112 }}>
                        <svg width={112} height={112} style={{ transform: "rotate(-90deg)" }}>
                            <circle cx={56} cy={56} r={48} fill="none" stroke={`${modelInfo.color}18`} strokeWidth={8} />
                            <circle cx={56} cy={56} r={48} fill="none" stroke={modelInfo.color} strokeWidth={8} strokeLinecap="round"
                                strokeDasharray={2 * Math.PI * 48}
                                strokeDashoffset={2 * Math.PI * 48 * (1 - progress / 100)}
                                style={{ transition: "stroke-dashoffset .12s linear", filter: `drop-shadow(0 0 5px ${modelInfo.color}99)` }}
                            />
                        </svg>
                        <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 1 }}>
                            <span style={{ fontSize: 26, fontWeight: 900, color: C.text, lineHeight: 1 }}>{pct}%</span>
                            <span style={{ fontSize: 9, color: C.sub, textTransform: "uppercase", letterSpacing: ".06em" }}>complete</span>
                        </div>
                    </div>
                    <div style={{ width: "100%" }}>
                        <div style={{ height: 5, background: `${modelInfo.color}15`, borderRadius: 3, overflow: "hidden" }}>
                            <div style={{ height: "100%", width: `${progress}%`, background: `linear-gradient(90deg,${modelInfo.color},${C.teal})`, transition: "width .12s linear", borderRadius: 3 }} />
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, fontSize: 10, color: C.sub }}>
                            <span>0%</span>
                            <span style={{ color: modelInfo.color, fontWeight: 700, fontFamily: "monospace", fontSize: 9 }}>{modelInfo.short}</span>
                            <span>100%</span>
                        </div>
                    </div>
                </div>

                {/* Inline warnings — shown after pipeline completes, inside the same view */}
                {warnings.length > 0 && pendingResults !== null && (
                    <div style={{ width: "100%", marginTop: 16, borderRadius: 14, border: "2px solid rgba(239,68,68,.5)", overflow: "hidden", background: C.surface }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "14px 18px", background: "rgba(239,68,68,.10)", borderBottom: "1px solid rgba(239,68,68,.25)" }}>
                            <AlertCircle size={16} color="#ef4444" />
                            <div style={{ flex: 1 }}>
                                <div style={{ fontSize: 13, fontWeight: 800, color: C.text }}>Some Resumes Could Not Be Read</div>
                                <div style={{ fontSize: 11, color: "#ef4444", marginTop: 1 }}>These files were skipped — screening completed on the remaining resumes</div>
                            </div>
                        </div>
                        <div style={{ padding: "12px 18px", display: "flex", flexDirection: "column", gap: 8 }}>
                            {warnings.map((w, i) => {
                                // Split into a header sentence + a list of filenames
                                const sep = w.lastIndexOf(': ');
                                const prefix = sep !== -1 ? w.slice(0, sep + 1) : w;   // e.g. "3 PDF(s) could not be parsed and were skipped"
                                const files  = sep !== -1 ? w.slice(sep + 2).split(', ').filter(Boolean) : [];
                                return (
                                    <div key={i} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                                        {/* Prefix / summary line */}
                                        <div style={{ fontSize: 12, color: C.text, fontWeight: 600, lineHeight: 1.5 }}>
                                            {prefix}
                                        </div>
                                        {/* One pill per skipped filename */}
                                        {files.length > 0 && (
                                            <div style={{ display: "flex", flexDirection: "column", gap: 5, marginTop: 2 }}>
                                                {files.map((f, fi) => (
                                                    <div key={fi} style={{
                                                        display: "flex", alignItems: "center", gap: 8,
                                                        padding: "6px 10px", borderRadius: 8,
                                                        background: "rgba(239,68,68,0.06)",
                                                        border: "1px solid rgba(239,68,68,0.18)",
                                                    }}>
                                                        <span style={{ fontSize: 11, lineHeight: 1, flexShrink: 0 }}>📄</span>
                                                        <span style={{ fontSize: 12, color: C.text, fontWeight: 600, wordBreak: "break-all" }}>{f}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                            <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.6, marginTop: 4 }}>
                                These resumes are usually password-protected, scanned without a text layer, or corrupted. The screening ran successfully on all other files.
                            </div>
                        </div>
                        <div style={{ padding: "10px 18px 14px" }}>
                            <button
                                onClick={() => onDoneRef.current(pendingResults)}
                                style={{ width: "100%", padding: "12px", borderRadius: 10, background: "rgba(239,68,68,.12)", border: "2px solid rgba(239,68,68,.45)", color: "#ef4444", fontSize: 13, fontWeight: 800, cursor: "pointer", fontFamily: "inherit", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                                <CheckCircle size={14} />
                                Continue to Results ({warnCountdown ?? 10}s)
                            </button>
                        </div>
                    </div>
                )}

                {/* Pipeline steps */}
                <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                    {PIPELINE_STEPS(config?.skillWeight || 55, MODELS[config?.model || "mpnet"]?.name || "MPNet").map((s, i) => {
                        const done = i < step;
                        const active = i === step;
                        return (
                            <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 13px", borderRadius: 10, background: done ? `${C.green}08` : active ? `${modelInfo.color}0f` : C.inputBg, border: `1px solid ${done ? `${C.green}20` : active ? `${modelInfo.color}26` : C.border}`, transition: "all .3s", position: "relative", overflow: "hidden" }}>
                                {/* Gradient left accent line for completed/active steps */}
                                {(done || active) && <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 3, borderRadius: "3px 0 0 3px", background: done ? C.green : modelInfo.color, transition: "background .3s" }} />}
                                <div style={{ width: 19, height: 19, borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", background: done ? `${C.green}18` : active ? `${modelInfo.color}18` : "transparent", border: `1.5px solid ${done ? C.green : active ? modelInfo.color : C.muted}` }}>
                                    {done ? <Check size={9} color={C.green} />
                                        : active ? <div style={{ width: 7, height: 7, borderRadius: "50%", border: `2px solid ${modelInfo.color}`, borderTopColor: "transparent", animation: "spin .7s linear infinite" }} />
                                            : <div style={{ width: 4, height: 4, borderRadius: "50%", background: C.muted }} />}
                                </div>
                                <div style={{ flex: 1 }}>
                                    <div style={{ fontSize: 10, fontFamily: "monospace", color: i <= step ? modelInfo.color : C.muted, fontWeight: 600, marginBottom: 1 }}>{s.label}</div>
                                    <div style={{ fontSize: 11, color: active ? C.cardText : C.muted }}>{s.desc}</div>
                                </div>
                                {done && <CheckCircle size={11} color={C.green} />}
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};

// ── Shared sub-components ────────────────────────────────────────────────────
// Declared before the views that use them to keep declaration order consistent
// with usage order and to avoid temporal dead zone confusion during analysis.

/**
 * FieldCard — glass-effect card with a label notched into its top border.
 * Used throughout Dashboard, Analytics, and Decisions view.
 *
 * Props:
 *   label    {string}  Section title rendered in the notch
 *   dot      {string}  Accent colour for the small indicator dot (optional)
 *   children {node}    Card body content
 *   xtra     {object}  Extra inline style overrides (optional)
 */
const FieldCard = ({ label, dot, children, xtra = {} }) => (
    <div style={{
        position: "relative",
        border: `1px solid ${C.border}`,
        borderRadius: 16,
        padding: "30px 20px 20px",
        background: C.surface,
        backdropFilter: "blur(14px)",
        WebkitBackdropFilter: "blur(14px)",
        ...xtra,
    }}>
        {/* Notched label sits on top of the border line */}
        <div style={{
            position: "absolute", top: -11, left: 16,
            display: "flex", alignItems: "center", gap: 6,
            background: C.bg, padding: "0 8px",
        }}>
            {dot && <div style={{ width: 6, height: 6, borderRadius: "50%", background: dot, flexShrink: 0 }} />}
            <span style={{ fontSize: 12, fontWeight: 800, color: C.text, letterSpacing: ".03em" }}>{label}</span>
        </div>
        {children}
    </div>
);

/**
 * SoftBar — horizontal progress bar with a glossy highlight layer.
 *
 * Props:
 *   pct   {number}  Fill percentage (0–100)
 *   color {string}  Bar fill colour
 *   h     {number}  Bar height in px (default 6)
 */
const SoftBar = ({ pct, color, h = 6 }) => (
    <div style={{ width: "100%", height: h, background: C.inputBg, borderRadius: h, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${Math.max(pct, 1)}%`, background: color, borderRadius: h, opacity: 0.85, transition: "width .8s cubic-bezier(.4,0,.2,1)", position: "relative" }}>
            {/* Glossy highlight stripe */}
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "40%", background: "rgba(255,255,255,0.18)", borderRadius: h }} />
        </div>
    </div>
);

/**
 * StatPill — compact KPI tile used in the Analytics summary strip.
 *
 * Props:
 *   label {string}  Metric name
 *   value {string}  Formatted value (e.g. '82%')
 *   color {string}  Accent colour
 *   sub   {string}  Optional secondary label
 */
const StatPill = ({ label, value, color, sub }) => (
    <div style={{ padding: "12px 14px", borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`, position: "relative", overflow: "hidden" }}>
        {/* Top accent gradient line */}
        <div style={{ position: "absolute", top: 0, left: "25%", right: "25%", height: 1, background: `linear-gradient(90deg,transparent,${color}55,transparent)` }} />
        <div style={{ fontSize: 20, fontWeight: 900, color, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>{value}</div>
        <div style={{ fontSize: 11, fontWeight: 600, color: C.text, marginTop: 4 }}>{label}</div>
        {sub && <div style={{ fontSize: 9, color: C.muted, marginTop: 1 }}>{sub}</div>}
    </div>
);

/**
 * LiquidTabBar — segmented control with an active-pill highlight.
 * Used in the Analytics view to switch between Overview / Decisions / Skills tabs.
 *
 * Props:
 *   sections {Array<{id, label}>}  Tab definitions
 *   active   {string}              Currently active tab id
 *   onChange {function}            Called with the new tab id on click
 */
const LiquidTabBar = ({ sections, active, onChange }) => (
    <div style={{
        display: "inline-flex", alignSelf: "flex-start",
        background: C.inputBg,
        backdropFilter: "blur(24px)", WebkitBackdropFilter: "blur(24px)",
        border: `1px solid ${C.border}`,
        borderRadius: 14, padding: 3, gap: 2,
        boxShadow: "0 2px 12px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.07)",
    }}>
        {sections.map(s => {
            const on = active === s.id;
            return (
                <button key={s.id} onClick={() => onChange(s.id)} style={{
                    padding: "6px 15px", borderRadius: 11, border: "none",
                    background: on ? C.surface : "transparent",
                    color: on ? C.text : C.sub,
                    fontSize: 12, fontWeight: on ? 700 : 500,
                    cursor: "pointer", fontFamily: "inherit",
                    transition: "all .18s cubic-bezier(.4,0,.2,1)",
                    whiteSpace: "nowrap",
                    boxShadow: on ? "0 1px 6px rgba(0,0,0,.15)" : "none",
                }}>{s.label}</button>
            );
        })}
    </div>
);

// ── Page views ────────────────────────────────────────────────────────────────

/**
 * DashboardView — landing page after a screening run.
 * Shows KPI strip, top-candidate card, score-distribution bar chart,
 * quick-action buttons, and a system info panel.
 *
 * Props:
 *   results      {Array}    Serialised candidate objects from the ML pipeline
 *   onNav        {function} Navigate to another tab by id
 *   isMobile     {boolean}  Responsive layout switch (< 768 px)
 *   activeModel  {string}   Currently selected embedding model key
 *   onModelChange{function} Switch active embedding model
 */
const DashboardView = ({ results, onNav, isMobile, activeModel, onModelChange }) => {
    if (!results || results.length === 0) {
        return (
            <EmptyState icon={LayoutDashboard} title="No Results Yet"
                sub="Upload resumes and run the screening pipeline to see ranked candidates here."
                action={
                    <button onClick={() => onNav("upload")} style={{ padding: "9px 20px", borderRadius: 10, border: "none", background: `linear-gradient(135deg,${C.blue},#6366f1)`, color: "#fff", cursor: "pointer", fontSize: 13, fontWeight: 700, fontFamily: "inherit", display: "flex", alignItems: "center", gap: 7 }}>
                        <Upload size={13} /> Upload Resumes
                    </button>
                }
            />
        );
    }

    const eligible = results.filter(c => c.eligible);
    const rejected = results.filter(c => !c.eligible);
    const top = [...eligible].sort((a, b) => (b.finalScore || 0) - (a.finalScore || 0))[0] || results[0];
    const avgScore = eligible.length ? Math.round(eligible.reduce((a, c) => a + (c.finalScore || 0), 0) / eligible.length * 100) : 0;
    const passRate = results.length ? Math.round(eligible.length / results.length * 100) : 0;
    const topScore = Math.round((top.finalScore || 0) * 100);

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>


            {/* KPI strip — four inline stat tiles */}
            <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr 1fr" : "repeat(4,1fr)", gap: 10 }}>
                {[
                    { icon: FileText, label: "Screened", value: results.length, color: C.blue, sub: "resumes processed" },
                    { icon: Award, label: "Shortlisted", value: eligible.length, color: C.green, sub: "eligible candidates" },
                    { icon: TrendingUp, label: "Avg Score", value: `${avgScore}%`, color: C.amber, sub: "across shortlisted" },
                    { icon: Star, label: "Pass Rate", value: `${passRate}%`, color: C.teal, sub: "of total pool" },
                ].map(({ icon: Icon, label, value, sub, color }) => (
                    <div key={label} style={{
                        padding: "14px 16px", borderRadius: 14, background: C.surface, border: `1px solid ${C.border}`, position: "relative", overflow: "hidden",
                        transition: "background .15s, border-color .15s"
                    }}
                        onMouseEnter={e => { e.currentTarget.style.background = `${color}08`; e.currentTarget.style.borderColor = `${color}28`; }}
                        onMouseLeave={e => { e.currentTarget.style.background = C.surface; e.currentTarget.style.borderColor = C.border; }}>
                        <div style={{ position: "absolute", top: 0, left: "20%", right: "20%", height: 1, background: `linear-gradient(90deg,transparent,${color}44,transparent)` }} />
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                            <div style={{ width: 30, height: 30, borderRadius: 8, background: `${color}12`, border: `1px solid ${color}20`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                                <Icon size={13} color={color} />
                            </div>
                            <span style={{ fontSize: 10, color: C.sub, fontWeight: 600 }}>{label}</span>
                        </div>
                        <div style={{ fontSize: 22, fontWeight: 900, color, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>{value}</div>
                        <div style={{ fontSize: 9, color: C.muted, marginTop: 3 }}>{sub}</div>
                    </div>
                ))}
            </div>

            {/* Top candidate + score distribution */}
            {eligible.length > 0 && (
                <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "360px 1fr", gap: 14, alignItems: "stretch" }}>

                    {/* Top candidate card — FIX: proper vertical rhythm + alignment */}
                    <FieldCard label="Top Candidate" dot={C.amber}>
                        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", padding: "8px 0 4px" }}>

                            {/* Gauge — centered with breathing room */}
                            <div style={{ marginBottom: 12 }}>
                                <GaugeArc value={topScore} />
                            </div>

                            {/* Name + Rank — clear separation from gauge */}
                            <div style={{ marginBottom: 16 }}>
                                <div style={{ fontSize: 15, fontWeight: 800, color: C.text, letterSpacing: "-0.01em" }}>{top.name}</div>
                                <div style={{ fontSize: 10, color: C.muted, marginTop: 4 }}>🥇 Rank #1</div>
                            </div>

                            {/* Score cards — equal width, balanced spacing */}
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, width: "100%", marginBottom: 16 }}>
                                {[["Skill", Math.round((top.skillScore || 0) * 100), C.blue], ["Semantic", Math.round((top.semanticScore || 0) * 100), C.teal]].map(([l, v, col]) => (
                                    <div key={l} style={{ padding: "12px 14px", borderRadius: 12, background: `${col}0a`, border: `1px solid ${col}20`, textAlign: "center" }}>
                                        <div style={{ fontSize: 18, fontWeight: 900, color: col, letterSpacing: "-0.02em" }}>{v}%</div>
                                        <div style={{ fontSize: 9, color: C.muted, marginTop: 3, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.04em" }}>{l}</div>
                                    </div>
                                ))}
                            </div>

                            {/* Matched skills — centered wrap with consistent gaps */}
                            {(top.matched_skills || []).length > 0 && (
                                <div style={{ display: "flex", flexWrap: "wrap", gap: 6, justifyContent: "center", width: "100%" }}>
                                    {(top.matched_skills || []).slice(0, 5).map(s => <SkillChip key={s} label={s} matched />)}
                                </div>
                            )}
                        </div>
                    </FieldCard>

                    {/* Score distribution — horizontal glass bars */}
                    <FieldCard label="Score Distribution" dot={C.blue}>
                        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 10 }}>
                            <button onClick={() => onNav("analytics")} style={{ fontSize: 10, color: C.blue, background: `${C.blue}10`, border: `1px solid ${C.blue}22`, borderRadius: 7, padding: "3px 10px", cursor: "pointer", fontFamily: "inherit", fontWeight: 600 }}>
                                Full Analytics →
                            </button>
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            {[...results].sort((a, b) => (b.finalScore || 0) - (a.finalScore || 0)).slice(0, 10).map((c, i) => {
                                const fs = Math.round((c.finalScore || 0) * 100);
                                const col = fs >= 70 ? C.green : fs >= 50 ? C.blue : C.amber;
                                const medals = ["🥇", "🥈", "🥉"];
                                return (
                                    <div key={c.id}
                                        title={`${c.name} — Final: ${fs}% · Skill: ${Math.round((c.skillScore || 0) * 100)}% · Semantic: ${Math.round((c.semanticScore || 0) * 100)}%`}
                                        style={{ display: "flex", alignItems: "center", gap: 9, borderRadius: 7, padding: "2px 4px", transition: "background .12s" }}
                                        onMouseEnter={e => e.currentTarget.style.background = C.tableFocus}
                                        onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                                        <div style={{ width: 20, textAlign: "center", flexShrink: 0, fontSize: i < 3 ? 13 : 9, color: C.muted, lineHeight: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
                                            {i < 3 ? medals[i] : i + 1}
                                        </div>
                                        <div style={{ width: 110, fontSize: 11, color: C.text, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flexShrink: 0 }}>{c.name}</div>
                                        <div style={{ flex: 1, height: 18, borderRadius: 6, overflow: "hidden", background: C.inputBg, border: `1px solid ${C.border}`, backdropFilter: "blur(6px)", position: "relative" }}>
                                            <div style={{ position: "absolute", top: 0, left: 0, bottom: 0, width: `${Math.max(fs, 1)}%`, background: `linear-gradient(90deg,${col}77,${col}cc)`, borderRadius: "6px 0 0 6px", transition: "width .8s cubic-bezier(.4,0,.2,1)" }}>
                                                <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "42%", background: "linear-gradient(180deg,rgba(255,255,255,0.16),rgba(255,255,255,0))", borderRadius: "6px 0 0 0" }} />
                                            </div>
                                            <div style={{ position: "relative", zIndex: 1, height: "100%", display: "flex", alignItems: "center", paddingLeft: 8 }}>
                                                <span style={{ fontSize: 9, fontWeight: 700, color: C.text, textShadow: "none" }}>{fs}%</span>
                                            </div>
                                        </div>
                                        <div style={{ width: 6, height: 6, borderRadius: "50%", flexShrink: 0, background: c.eligible ? C.green : "#ef4444", opacity: .7 }} />
                                    </div>
                                );
                            })}
                        </div>
                        <div style={{ display: "flex", gap: 14, marginTop: 12, paddingTop: 10, borderTop: `1px solid ${C.border}` }}>
                            {[["● Shortlisted", C.green], ["● Rejected", "#ef4444"]].map(([l, c]) => (
                                <div key={l} style={{ fontSize: 9, color: C.muted }}><span style={{ color: c }}>{l.slice(0, 1)}</span>{l.slice(1)}</div>
                            ))}
                        </div>
                    </FieldCard>
                </div>
            )}

            {/* Quick actions — glass pill row */}
            <div style={{ borderRadius: 16, border: `1px solid ${C.border}`, background: C.surface, backdropFilter: "blur(14px)", padding: "14px 18px" }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: C.sub, marginBottom: 12 }}>Quick Actions</div>
                <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr 1fr" : "repeat(4,1fr)", gap: 8 }}>
                    {[
                        { label: "View Rankings", icon: Users, nav: "candidates", color: C.blue },
                        { label: "Analytics", icon: BarChart2, nav: "analytics", color: C.teal },
                        { label: "Job Config", icon: Briefcase, nav: "config", color: C.amber },
                        { label: "New Screening", icon: RefreshCw, nav: "upload", color: C.pink },
                    ].map(({ label, icon: Icon, nav: n, color }) => (
                        <button key={n} onClick={() => onNav(n)} style={{
                            padding: "11px 14px", borderRadius: 12, border: `1px solid ${color}22`,
                            background: `${color}08`, color, fontSize: 12, fontWeight: 600,
                            cursor: "pointer", display: "flex", alignItems: "center", gap: 7,
                            fontFamily: "inherit", transition: "all .15s",
                        }}
                            onMouseEnter={e => { e.currentTarget.style.background = `${color}14`; e.currentTarget.style.borderColor = `${color}38`; }}
                            onMouseLeave={e => { e.currentTarget.style.background = `${color}08`; e.currentTarget.style.borderColor = `${color}22`; }}>
                            <Icon size={13} /> {label}
                        </button>
                    ))}
                </div>
            </div>


            {/* About This System */}
            <FieldCard label="About This System" dot={C.blue}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 18 }}>
                    <div style={{ width: 40, height: 40, borderRadius: 12, background: `linear-gradient(135deg,${C.blue},#a78bfa)`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                        <Sparkles size={18} color="#fff" />
                    </div>
                    <div>
                        <div style={{ fontSize: 15, fontWeight: 800, color: C.text }}>ML-Based Resume Screening System</div>
                        <div style={{ fontSize: 11, color: C.sub }}>By <span style={{ color: C.blue, fontWeight: 700 }}>Madhan Kumar</span> · B.Sc Computer Science · Final Year Project</div>
                    </div>
                </div>

                {/* Pipeline steps — accurate descriptions per active model */}
                <div style={{ marginBottom: 20 }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".07em", marginBottom: 10 }}>How it works — 5 step pipeline</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        {[
                            {
                                step: "1", label: "Upload PDFs", color: C.blue,
                                desc: "Resume PDF files are uploaded to a temporary session folder on the Flask server. Text is extracted using pdfplumber, with PyMuPDF (fitz) as a fallback for scanned or complex PDFs.",
                            },
                            {
                                step: "2", label: "Extract Info", color: C.teal,
                                desc: "Candidate name, email, phone, LinkedIn/GitHub links, skills, and experience years are extracted from raw text using regex patterns and a curated 150+ skill keyword database.",
                            },
                            {
                                step: "3", label: "Semantic Embedding", color: "#a78bfa",
                                desc: `Resume text and job description are independently encoded into ${activeModel === "mxbai" ? "1024" : "768"}-dimensional dense vectors using ${MODELS[activeModel]?.name || "MPNet"} (${MODELS[activeModel]?.short || "sentence-transformers/all-mpnet-base-v2"}). The model captures semantic meaning, not just keyword overlap.`,
                            },
                            {
                                step: "4", label: "Cosine Similarity", color: C.amber,
                                desc: "Two-pass cosine similarity: 40% weight on the full resume embedding + 60% weight on key-section embeddings (Skills, Experience, Projects blocks). Final semantic score is sigmoid-calibrated and dampened by JD quality factor.",
                            },
                            {
                                step: "5", label: "Weighted Scoring & Ranking", color: C.green,
                                desc: (() => {
                                    // Show the actual weights used in the last run if available
                                    const r0 = results?.[0];
                                    const sw = r0?.skill_weight != null
                                        ? Math.round(r0.skill_weight * 100)
                                        : r0 != null ? 55 : null;
                                    const weightStr = sw != null
                                        ? `This batch used ${sw}% skill / ${100 - sw}% semantic.`
                                        : "Default: 55% skill / 45% semantic — configurable in the Upload tab.";
                                    return `Final Score = (Skill Weight × Skill Score) + (Semantic Weight × Semantic Score). ${weightStr} Shortlist threshold = 60th percentile of the batch's final scores.`;
                                })(),
                            },
                        ].map(({ step, label, desc, color }) => (
                            <div key={step} style={{ display: "flex", gap: 12, padding: "10px 14px", borderRadius: 10, background: `${color}12`, border: `1px solid ${color}24` }}>
                                <div style={{ width: 24, height: 24, borderRadius: 7, background: `${color}16`, border: `1px solid ${color}28`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 900, color, flexShrink: 0, marginTop: 1 }}>{step}</div>
                                <div>
                                    <div style={{ fontSize: 12, fontWeight: 700, color: C.text, marginBottom: 2 }}>{label}</div>
                                    <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.6 }}>{desc}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Models */}
                <div style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".07em", marginBottom: 10 }}>Embedding models</div>
                <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "repeat(3,1fr)", gap: 10 }}>
                    {Object.values(MODELS).map(m => (
                        <div key={m.key} style={{ padding: "12px 14px", borderRadius: 12, background: `${m.color}12`, border: `1px solid ${m.color}${activeModel === m.key ? "40" : "22"}`, position: "relative", transition: "border-color .15s" }}
                            onMouseEnter={e => e.currentTarget.style.borderColor = `${m.color}38`}
                            onMouseLeave={e => e.currentTarget.style.borderColor = `${m.color}${activeModel === m.key ? "40" : "18"}`}>
                            {activeModel === m.key && (
                                <div style={{ position: "absolute", top: 8, right: 8, padding: "1px 7px", borderRadius: 10, fontSize: 9, fontWeight: 700, background: `${m.color}20`, color: m.color }}>● Active</div>
                            )}
                            <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 4 }}>
                                <div style={{ width: 7, height: 7, borderRadius: "50%", background: m.color, flexShrink: 0 }} />
                                <span style={{ fontSize: 12, fontWeight: 800, color: C.text }}>{m.name}</span>
                                <span style={{ fontSize: 8, padding: "1px 6px", borderRadius: 10, fontWeight: 700, background: `${m.color}16`, color: m.color }}>{m.badge}</span>
                            </div>
                            <div style={{ fontSize: 9, fontFamily: "monospace", color: C.muted, marginBottom: 4, wordBreak: "break-all" }}>{m.short}</div>
                            <div style={{ fontSize: 10, color: C.text, lineHeight: 1.5 }}>{m.detail}</div>
                        </div>
                    ))}
                </div>
            </FieldCard>

        </div>
    );
};

// Medal colour palette for top-3 ranked candidates (gold / silver / bronze).
const RANK_COLORS = {
    1: { bg: "rgba(255,215,0,.18)", border: "rgba(255,215,0,.45)", text: "#FFD700" },
    2: { bg: "rgba(192,192,192,.18)", border: "rgba(192,192,192,.45)", text: "#C0C0C0" },
    3: { bg: "rgba(205,127,50,.18)", border: "rgba(205,127,50,.45)", text: "#CD7F32" },
};

/**
 * downloadCSV — builds and triggers a browser download for a CSV file.
 * Used by both the Candidates view and the Analytics Decisions tab.
 *
 * @param {string[]} headers  Column header names
 * @param {string[][]} rows   One array per candidate, values pre-formatted
 * @param {string} filename   File name (including .csv extension)
 */
function downloadCSV(headers, rows, filename) {
    const csv = [
        headers.join(","),
        ...rows.map(r => r.join(",")),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    // Release the object URL to free memory
    URL.revokeObjectURL(url);
}

/**
 * csvField — wraps a value in double-quotes and escapes any embedded quotes
 * with single-quotes so the CSV remains valid if names/emails contain commas.
 */
const csvField = (v) => `"${String(v || "").replace(/"/g, "'")}"`;

/**
 * exportDecisionGroup — downloads a CSV of a shortlisted candidate group with
 * full scoring breakdown. Called from the Analytics → Decisions tab.
 *
 * @param {object[]} list       Candidate result objects to export
 * @param {string}   groupLabel Human-readable group name (used as file prefix)
 */
function exportDecisionGroup(list, groupLabel) {
    if (!list?.length) return;
    const headers = ["Rank", "Name", "Email", "Phone", "Final Score", "Skill Score", "Semantic Score"];
    const rows = list.map(c => [
        csvField(c.rank || "—"),
        csvField(c.name),
        csvField(c.email),
        csvField(c.phone),
        `${Math.round((c.finalScore || 0) * 100)}%`,
        `${Math.round((c.skillScore || 0) * 100)}%`,
        `${Math.round((c.semanticScore || 0) * 100)}%`,
    ]);
    downloadCSV(headers, rows, groupLabel.toLowerCase().replace(/\s+/g, "_") + "_candidates.csv");
}

/**
 * exportToCSV — downloads a lightweight CSV (rank, name, email, phone) of all
 * shortlisted candidates. Called from the Candidates view export button.
 */
function exportToCSV(results) {
    const shortlisted = results.filter(c => c.eligible);
    if (!shortlisted.length) return;
    const headers = ["Rank", "Name", "Email", "Phone"];
    const rows = shortlisted.map(c => [
        csvField(c.rank),
        csvField(c.name),
        csvField(c.email),
        csvField(c.phone),
    ]);
    downloadCSV(headers, rows, "shortlisted_candidates.csv");
}

/**
 * CandidatesView — ranked candidate table with filterable rows (All / Shortlisted / Rejected).
 * Clicking a row opens the Drawer panel with full candidate detail and skill radar.
 *
 * Props:
 *   results  {Array}    Serialised candidate objects from the ML pipeline
 *   onNav    {function} Navigate to another tab (e.g. redirect to Upload when empty)
 *   isMobile {boolean}  Responsive layout switch (< 768 px)
 */
const CandidatesView = ({ results, onNav, isMobile }) => {
    const [selected, setSelected] = useState(null);
    const [filter, setFilter] = useState("All");

    if (!results || results.length === 0) {
        return (
            <EmptyState icon={Users} title="No Candidates Yet"
                sub="Run the screening pipeline to see your ranked candidates here."
                action={
                    <button onClick={() => onNav("upload")} style={{ padding: "9px 20px", borderRadius: 10, border: "none", background: `linear-gradient(135deg,${C.blue},#6366f1)`, color: "#fff", cursor: "pointer", fontSize: 13, fontWeight: 700, fontFamily: "inherit", display: "flex", alignItems: "center", gap: 7 }}>
                        <Zap size={13} /> Start Screening
                    </button>
                }
            />
        );
    }

    const filtered = filter === "All"
        ? results
        : filter === "Shortlisted"
            ? results.filter(c => c.eligible)
            : results.filter(c => !c.eligible);

    return (
        <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: isMobile ? "flex-start" : "center", flexWrap: "wrap", gap: 11, marginBottom: 16 }}>
                <div>
                    <h2 style={{ fontSize: 20, fontWeight: 800, color: C.text, margin: 0 }}>Candidate Rankings</h2>
                    <p style={{ color: C.sub, marginTop: 3, fontSize: 12 }}>Click a candidate's row to view their full profile, scores, and skill gap analysis.</p>
                </div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {["All", "Shortlisted", "Rejected"].map(f => (
                        <button key={f} onClick={() => setFilter(f)} style={{ padding: "5px 12px", borderRadius: 20, fontSize: 11, fontWeight: 600, cursor: "pointer", background: filter === f ? `${C.blue}20` : C.inputBg, color: filter === f ? C.blue : C.sub, border: `1px solid ${filter === f ? `${C.blue}40` : C.border}`, fontFamily: "inherit" }}>{f}</button>
                    ))}
                    <button onClick={() => exportToCSV(results)} style={{ padding: "5px 12px", borderRadius: 20, fontSize: 11, fontWeight: 600, cursor: "pointer", background: `${C.green}12`, color: C.green, border: `1px solid ${C.green}30`, display: "flex", alignItems: "center", gap: 5, fontFamily: "inherit" }}>
                        <Download size={10} /> Export Shortlisted
                    </button>
                </div>
            </div>

            <div style={card({ padding: 0, overflow: "hidden" })}>
                <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 560 }}>
                        <thead>
                            <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                                {["Rank", "Candidate", "Skill", "Semantic", "Final", "Status", ""].map(h => (
                                    <th key={h} style={{ padding: "10px 15px", textAlign: "left", fontSize: 10, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".07em", whiteSpace: "nowrap" }}>{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {filtered.map((c, i) => (
                                <tr key={c.id} onClick={() => setSelected(c)}
                                    style={{ borderBottom: `1px solid ${C.border}22`, cursor: "pointer", transition: "all .18s ease" }}
                                    onMouseEnter={e => {
                                        e.currentTarget.style.background = C.tableFocus;
                                        e.currentTarget.style.transform = "translateY(-1px)";
                                        e.currentTarget.style.boxShadow = `0 2px 8px rgba(0,0,0,.08)`;
                                        const tds = e.currentTarget.querySelectorAll("td");
                                        const chevron = tds[tds.length - 1]?.querySelector("svg");
                                        if (chevron) chevron.style.transform = "translateX(3px)";
                                    }}
                                    onMouseLeave={e => {
                                        e.currentTarget.style.background = "transparent";
                                        e.currentTarget.style.transform = "none";
                                        e.currentTarget.style.boxShadow = "none";
                                        const tds = e.currentTarget.querySelectorAll("td");
                                        const chevron = tds[tds.length - 1]?.querySelector("svg");
                                        if (chevron) chevron.style.transform = "translateX(0)";
                                    }}>
                                    <td style={{ padding: "12px 15px" }}>
                                        {c.eligible ? (() => {
                                            const medal = RANK_COLORS[c.rank] || { bg: C.inputBg, border: C.border, text: C.sub };
                                            return (
                                                <span style={{ width: 27, height: 27, borderRadius: 7, display: "inline-flex", alignItems: "center", justifyContent: "center", background: medal.bg, border: `1.5px solid ${medal.border}`, fontSize: 11, fontWeight: 800, color: medal.text, boxShadow: c.rank <= 3 ? `0 0 8px ${medal.text}44` : "none" }}>
                                                    {c.rank <= 3 ? ["🥇", "🥈", "🥉"][c.rank - 1] : c.rank}
                                                </span>
                                            );
                                        })() : <span style={{ fontSize: 12, color: C.muted }}>—</span>}
                                    </td>
                                    <td style={{ padding: "12px 15px" }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                                            <div style={{ width: 30, height: 30, borderRadius: 8, background: `linear-gradient(135deg,hsl(${(c.id || 1) * 47},55%,28%),hsl(${(c.id || 1) * 47},55%,38%))`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 800, color: `hsl(${(c.id || 1) * 47},80%,80%)`, flexShrink: 0 }}>
                                                {(c.name || "?").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
                                            </div>
                                            <div>
                                                <div style={{ fontSize: 13, fontWeight: 600, color: C.text, whiteSpace: "nowrap" }}>{c.name || "Unknown"}</div>
                                                <div style={{ fontSize: 11, color: c.email ? C.sub : C.muted }}>{c.email || "Not Found"}</div>
                                            </div>
                                        </div>
                                    </td>
                                    <td style={{ padding: "12px 15px" }}><CircularRing value={Math.round((c.skillScore || 0) * 100)} size={38} sw={3} color={C.blue} /></td>
                                    <td style={{ padding: "12px 15px" }}><CircularRing value={Math.round((c.semanticScore || 0) * 100)} size={38} sw={3} color={C.teal} /></td>
                                    <td style={{ padding: "12px 15px" }}><CircularRing value={Math.round((c.finalScore || 0) * 100)} size={38} sw={3} color={C.amber} /></td>
                                    <td style={{ padding: "12px 15px" }}><Badge status={c.eligible ? "Shortlisted" : "Rejected"} /></td>
                                    <td style={{ padding: "12px 15px" }}>
                                        <ChevronRight size={13} color={C.sub} style={{ transition: "transform .15s", display: "block" }} />
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            <Drawer candidate={selected} onClose={() => setSelected(null)} isMobile={isMobile} />
        </div>
    );
};

/**
 * JobConfigView — persistent job profile editor.
 * Saves job description, required skills, and scoring weights to config.json
 * on the backend so they auto-fill the Upload form on every future session.
 * No props — reads and writes via GET/POST /api/config.
 */
const JobConfigView = ({ isMobile }) => {
    const [saved, setSaved] = useState(false);
    const [saveError, setSaveError] = useState(null);
    const [cfg, setCfg] = useState({ jd: "", skills: [], minExp: 0, skillW: 55, semanticW: 45 });
    const [newSkill, setNewSkill] = useState("");

    // Load the saved job configuration from backend on mount
    useEffect(() => {
        fetch(`${BASE}/api/config`)
            .then(r => r.json())
            .then(data => {
                if (data) setCfg(prev => ({
                    ...prev,
                    jd: data.job_description || "",
                    skills: data.required_skills || [],
                    skillW: data.scoring?.skill_weight ? Math.round(data.scoring.skill_weight * 100) : 55,
                    semanticW: data.scoring?.semantic_weight ? Math.round(data.scoring.semantic_weight * 100) : 45,
                    minExp: data.scoring?.min_experience_years || 0,
                }));
            })
            .catch(() => { /* silently ignore if backend is offline */ });
    }, []); // BASE is a module-level constant — intentional empty deps

    const addCfgSkill = (raw = newSkill) => {
        const parts = raw.split(",").map(s => s.trim()).filter(s => s.length > 0);
        const toAdd = [], invalid = [];
        parts.forEach(p => {
            const err = validateSkillInput(p);
            if (err) invalid.push(err);
            else toAdd.push(p.toLowerCase());
        });
        if (invalid.length) setSaveError(invalid[0]);
        if (toAdd.length) setCfg(prev => { const ex = new Set(prev.skills); return { ...prev, skills: [...prev.skills, ...toAdd.filter(s => !ex.has(s))] }; });
        setNewSkill("");
    };

    // POST the current config to Flask; show a "Saved" confirmation on success
    const save = async () => {
        const jdQ = validateJD(cfg.jd);
        if (jdQ.score === 0) {
            setSaveError(`Cannot save: ${jdQ.reason}`);
            setTimeout(() => setSaveError(null), 5000);
            return;
        }
        setSaveError(null);
        try {
            const res = await fetch(`${BASE}/api/config`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    job_description: cfg.jd,
                    required_skills: cfg.skills,
                    scoring: {
                        skill_weight: cfg.skillW / 100,
                        semantic_weight: cfg.semanticW / 100,
                        min_experience_years: cfg.minExp,
                        top_n: 100,
                    },
                }),
            });
            if (res.ok) {
                setSaved(true);
                setTimeout(() => setSaved(false), 2500);
            } else {
                setSaveError("Failed to save — is the Flask backend running?");
                setTimeout(() => setSaveError(null), 5000);
            }
        } catch {
            setSaveError("Cannot reach backend — start Flask with: python app.py");
            setTimeout(() => setSaveError(null), 5000);
        }
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Header + What is this */}
            <div>
                <h2 style={{ fontSize: 20, fontWeight: 800, color: C.text, margin: 0 }}>Job Configuration</h2>
                <p style={{ color: C.sub, marginTop: 4, fontSize: 12 }}>
                    Saved values are written to <code style={{ color: C.teal, background: `${C.teal}15`, padding: "1px 6px", borderRadius: 4, fontSize: 11 }}>config.json</code> and become the default pre-fill for every new Upload session.
                </p>
            </div>

            {/* What is this used for */}
            <div style={card({ background: `${C.blue}06`, borderColor: `${C.blue}22`, padding: 16 })}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 9 }}>
                    <Info size={13} color={C.blue} />
                    <span style={{ fontSize: 12, fontWeight: 700, color: C.blue }}>What is this page for?</span>
                </div>
                <p style={{ fontSize: 12, color: C.cardText, lineHeight: 1.7, margin: 0 }}>
                    Think of this as your <strong style={{ color: C.text }}>saved job profile</strong>. When screening multiple resume batches for the same role, you don't need to retype the job description and required skills each time — save them here once and they will auto-fill on the Upload tab. You can also standardise the scoring weights (skill vs. semantic) for consistent results across sessions.
                </p>
            </div>

            {/* Two-column layout same as Upload */}
            <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr", gap: 18, alignItems: "stretch" }}>
                {/* Left: JD */}
                <div style={{ ...card(), display: "flex", flexDirection: "column" }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 6, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 7 }}>
                        <span style={{ display: "flex", alignItems: "center", gap: 7 }}><Briefcase size={12} /> Job Description</span>
                        {cfg.jd.trim() && (() => { const q = validateJD(cfg.jd); return (
                            <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 10px", borderRadius: 20, background: `${q.color}18`, color: q.color, border: `1px solid ${q.color}30` }}>{q.label}</span>
                        ); })()}
                    </div>
                    {cfg.jd.trim() && (() => { const q = validateJD(cfg.jd); return q.score < 3 ? (
                        <div style={{ fontSize: 11, color: q.color, marginBottom: 8, display: "flex", gap: 5, alignItems: "flex-start", lineHeight: 1.5 }}>
                            <AlertCircle size={12} style={{ flexShrink: 0, marginTop: 1 }} />{q.reason}
                        </div>
                    ) : null; })()}
                    <textarea value={cfg.jd} onChange={e => { setCfg(p => ({ ...p, jd: e.target.value })); e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; }}
                        placeholder="Describe what the candidate will be doing, required responsibilities, and what you expect from them. Be specific for better results."
                        style={{ flex: 1, width: "100%", background: C.inputBg, border: `1px solid ${C.border}`, borderRadius: 10, color: C.cardText, fontSize: 13, padding: "10px 12px", resize: "none", outline: "none", fontFamily: "inherit", lineHeight: 1.6, boxSizing: "border-box", overflow: "hidden", minHeight: 160 }} />
                </div>

                {/* Right: Skills + Weights */}
                <div style={card()}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 14, display: "flex", alignItems: "center", gap: 7 }}>
                        <Target size={12} /> Skills & Weights
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 7 }}>
                        <label style={{ fontSize: 12, color: C.sub }}>Required Skills</label>
                        {cfg.skills.length > 0 && (
                            <button onClick={() => setCfg(p => ({ ...p, skills: [] }))} style={{ fontSize: 11, color: "#ef4444", background: "rgba(239,68,68,.08)", border: "1px solid rgba(239,68,68,.2)", borderRadius: 6, padding: "2px 9px", cursor: "pointer", fontFamily: "inherit", fontWeight: 600 }}>Clear All</button>
                        )}
                    </div>
                    {cfg.skills.length > 0 && (
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                            {cfg.skills.map(s => (
                                <div key={s} style={{ display: "flex", alignItems: "center", gap: 5, padding: "3px 9px", borderRadius: 20, background: `${C.blue}18`, border: `1px solid ${C.blue}30`, color: C.blue, fontSize: 12, fontWeight: 500 }}>
                                    {s}
                                    <button onClick={() => setCfg(p => ({ ...p, skills: p.skills.filter(x => x !== s) }))} style={{ background: "none", border: "none", cursor: "pointer", color: C.sub, padding: 0, display: "flex" }}><X size={9} /></button>
                                </div>
                            ))}
                        </div>
                    )}
                    <div style={{ display: "flex", gap: 7, marginBottom: 18 }}>
                        <input value={newSkill} onChange={e => setNewSkill(e.target.value)}
                            onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); addCfgSkill(); } }}
                            placeholder="e.g. python, sql, machine learning — press Enter to add"
                            style={{ flex: 1, background: C.inputBg, border: `1px solid ${C.border}`, borderRadius: 8, color: C.cardText, fontSize: 13, padding: "8px 11px", outline: "none", fontFamily: "inherit" }} />
                        <button onClick={() => addCfgSkill()} style={{ padding: "8px 13px", borderRadius: 8, background: `${C.blue}18`, border: `1px solid ${C.blue}30`, color: C.blue, cursor: "pointer", fontFamily: "inherit", fontWeight: 700 }}>
                            <Plus size={13} />
                        </button>
                    </div>

                    {[
                        { label: "Min Experience", key: "minExp", min: 0, max: 10, step: .5, color: C.amber, fmt: v => v === 0 ? "Freshers — 0 yrs" : `${v}+ yrs` },
                        { label: "Skill Weight", key: "skillW", min: 0, max: 100, step: 5, color: C.blue, fmt: v => `${v}% skill / ${100 - v}% semantic` },
                    ].map(({ label, key, min, max, step, color, fmt }) => (
                        <div key={key} style={{ marginBottom: 14 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                                <span style={{ fontSize: 12, color: C.sub }}>{label}</span>
                                <span style={{ fontSize: 12, color, fontWeight: 700 }}>{fmt(cfg[key])}</span>
                            </div>
                            <input type="range" min={min} max={max} step={step} value={cfg[key]}
                                onChange={e => setCfg(p => ({ ...p, [key]: parseFloat(e.target.value), ...(key === "skillW" ? { semanticW: 100 - parseFloat(e.target.value) } : {}) }))}
                                style={{ width: "100%", accentColor: color, cursor: "pointer" }} />
                        </div>
                    ))}
                </div>
            </div>

            {/* Save error banner */}
            {saveError && (
                <div style={{ display: "flex", alignItems: "center", gap: 9, padding: "10px 16px", borderRadius: 10, background: "rgba(239,68,68,.08)", border: "1px solid rgba(239,68,68,.22)", fontSize: 12, color: "#ef4444" }}>
                    <AlertCircle size={13} style={{ flexShrink: 0 }} />
                    {saveError}
                </div>
            )}

            {/* Centered save button */}
            <div style={{ display: "flex", justifyContent: "center" }}>
                <button onClick={save} style={{ padding: "12px 36px", borderRadius: 12, border: `1px solid ${saved ? `${C.green}40` : `${C.blue}40`}`, background: saved ? `${C.green}12` : `linear-gradient(135deg,${C.blue}18,#a78bfa18)`, color: saved ? C.green : C.blue, cursor: "pointer", fontSize: 14, fontWeight: 700, display: "flex", alignItems: "center", gap: 9, fontFamily: "inherit", transition: "all .3s", boxShadow: saved ? `0 0 16px ${C.green}22` : `0 0 16px ${C.blue}18` }}>
                    {saved ? <><CheckCircle size={15} /> Saved to config.json</> : <><Save size={15} /> Save Configuration</>}
                </button>
            </div>
        </div>
    );
};

const ModelEvalMetrics = ({ results, activeModel, onNavDecisions }) => {
    const ev = useMemo(() => {
        if (!results || results.length === 0) return null;

        // Threshold = 60th percentile of batch scores, clamped to model default ±10%
        const modelDefault = MODEL_THRESH[activeModel] || 0.50;
        const sortedFinals = [...results].map(c => c.finalScore || 0).sort((a, b) => a - b);
        const p60idx = Math.max(Math.floor(sortedFinals.length * 0.60) - 1, 0);
        const p60 = sortedFinals[Math.min(p60idx, sortedFinals.length - 1)];
        const THRESH = Math.max(modelDefault - 0.10, Math.min(modelDefault + 0.10, p60));

        let tp = 0, tn = 0, fp = 0, fn = 0;
        const sims = [], finals = [];
        const bands = { "Strong Fit": 0, "Borderline": 0, "Weak Fit": 0 };

        results.forEach(c => {
            const sem = c.semanticScore || 0;
            const final = c.finalScore || 0;
            sims.push(sem); finals.push(final);

            // Use backend band — matches the fixed band names: 'Strong Fit', 'Borderline', 'Weak Fit'
            const band = c.band || (c.eligible ? "Borderline" : "Weak Fit");
            if (band === "Strong Fit") bands["Strong Fit"]++;
            else if (band === "Borderline") bands["Borderline"]++;
            else bands["Weak Fit"]++;

            const pred = final >= THRESH ? 1 : 0;
            const actual = c.eligible ? 1 : 0;
            if (pred === 1 && actual === 1) tp++;
            else if (pred === 0 && actual === 0) tn++;
            else if (pred === 1 && actual === 0) fp++;
            else fn++;
        });

        const n = results.length;
        const tot = tp + tn + fp + fn;
        const acc = tot > 0 ? (tp + tn) / tot : null;
        const prec = (tp + fp) > 0 ? tp / (tp + fp) : null;
        const rec = (tp + fn) > 0 ? tp / (tp + fn) : null;
        const f1 = prec != null && rec != null && (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : null;

        const fnRecovered = results.filter(c => c.fn_recovered).length; // Candidates rescued by semantic score

        return { tp, tn, fp, fn, tot, acc, prec, rec, f1, THRESH, bands, fnRecovered };
    }, [results, activeModel]);

    if (!ev) return null;

    const pct = v => v != null ? Math.round(v * 100) + "%" : "—";
    const d4 = v => v != null ? v.toFixed(4) : "—";
    const hue = v => v == null ? C.muted : v >= 0.80 ? C.green : v >= 0.60 ? C.amber : "#ef4444";
    const mc = ({ mpnet: "#6366f1", mxbai: C.teal, arctic: "#f472b6" })[activeModel] || C.blue;

    const bar = v => v != null ? Math.round(v * 100) : 0;

    return (
        <FieldCard label="Model Evaluation Metrics" dot={C.teal} xtra={{ padding: "24px" }}>
            {/* ── Row 1: header strip ─────────────────────────────── */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14, gap: 8 }}>
                <div style={{ fontSize: 11, color: C.sub, lineHeight: 1.5 }}>
                    Fit threshold: <strong style={{ color: C.text }}>Final ≥ {Math.round(ev.THRESH * 100)}%</strong>
                    <span style={{ color: C.muted }}> · 60th pct of this batch · model default {Math.round((MODEL_THRESH[activeModel] || 0.5) * 100)}% ± 10%</span>
                    {ev.fnRecovered > 0 && <span style={{ color: C.amber, fontWeight: 700 }}> · ↑ {ev.fnRecovered} rescued by semantic score</span>}
                </div>
                <span style={{ fontSize: 9, padding: "2px 9px", borderRadius: 20, fontWeight: 800, background: `${mc}14`, color: mc, border: `1px solid ${mc}28`, textTransform: "uppercase", letterSpacing: ".06em", flexShrink: 0 }}>{activeModel || "—"} · n={ev.tot}</span>
            </div>

            {/* ── Row 2: 4 compact metric bars side by side ────────── */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8, marginBottom: 14 }}>
                {[
                    { label: "Accuracy", val: ev.acc, formula: "(TP+TN) / n", desc: "Correctly classified" },
                    { label: "Precision", val: ev.prec, formula: "TP / (TP+FP)", desc: "Of shortlisted, truly fit" },
                    { label: "Recall", val: ev.rec, formula: "TP / (TP+FN)", desc: "Of fit, actually caught" },
                    { label: "F1 Score", val: ev.f1, formula: "2·P·R / (P+R)", desc: "Balance of P & R" },
                ].map(({ label, val, formula, desc }) => {
                    const clr = hue(val);
                    const w = bar(val);
                    return (
                        <div key={label} style={{ borderRadius: 10, border: `1px solid ${clr}22`, overflow: "hidden", background: `${clr}05` }}>
                            {/* fill bar at top */}
                            <div style={{ height: 3, background: C.inputBg, position: "relative" }}>
                                <div style={{ position: "absolute", inset: 0, width: `${w}%`, background: `linear-gradient(90deg,${clr}88,${clr})`, borderRadius: 3, transition: "width .9s cubic-bezier(.4,0,.2,1)" }} />
                            </div>
                            <div style={{ padding: "8px 10px" }}>
                                <div style={{ display: "flex", alignItems: "baseline", gap: 5, marginBottom: 2 }}>
                                    <span style={{ fontSize: 20, fontWeight: 900, color: clr, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>{pct(val)}</span>
                                    <span style={{ fontSize: 8, fontFamily: "monospace", color: `${clr}99` }}>{d4(val)}</span>
                                </div>
                                <div style={{ fontSize: 9, fontWeight: 700, color: C.text, textTransform: "uppercase", letterSpacing: ".05em" }}>{label}</div>
                                <div style={{ fontSize: 8, color: C.muted, marginTop: 2 }}>{desc}</div>
                                <div style={{ fontSize: 8, fontFamily: "monospace", color: `${clr}70`, marginTop: 3 }}>{formula}</div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* ── Row 3: confusion matrix + soft bands ──── */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>

                {/* Left: Confusion Matrix 2×2 */}
                <div style={{ borderRadius: 10, border: `1px solid ${C.border}`, overflow: "hidden" }}>
                    <div style={{ padding: "7px 13px", background: C.inputBg, borderBottom: `1px solid ${C.border}` }}>
                        <span style={{ fontSize: 9, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".06em" }}>Confusion Matrix</span>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gridTemplateRows: "1fr 1fr" }}>
                        {[
                            { ab: "TP", label: "True Positives", color: C.green, val: ev.tp, verdict: "Predicted Fit → Actually Fit ✓" },
                            { ab: "TN", label: "True Negatives", color: C.teal, val: ev.tn, verdict: "Predicted Not Fit → Actually Not Fit ✓" },
                            { ab: "FP", label: "False Positives", color: C.amber, val: ev.fp, verdict: "Predicted Fit → Actually Not Fit ✗" },
                            { ab: "FN", label: "False Negatives", color: "#ef4444", val: ev.fn, verdict: "Predicted Not Fit → Actually Fit ✗" },
                        ].map(({ ab, label, color, val, verdict }, i) => (
                            <div key={ab} style={{
                                padding: "14px 14px",
                                borderRight: i % 2 === 0 ? `1px solid ${C.border}` : "none",
                                borderBottom: i < 2 ? `1px solid ${C.border}` : "none",
                                display: "flex", alignItems: "center", gap: 11,
                            }}>
                                <div style={{ width: 34, height: 34, borderRadius: 8, background: `${color}12`, border: `1px solid ${color}28`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                                    <span style={{ fontSize: 10, fontFamily: "monospace", fontWeight: 900, color }}>{ab}</span>
                                </div>
                                <div>
                                    <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginBottom: 4 }}>
                                        <span style={{ fontSize: 20, fontWeight: 900, color, fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>{val}</span>
                                        <span style={{ fontSize: 10, fontWeight: 700, color: C.text, lineHeight: 1 }}>{label}</span>
                                    </div>
                                    <div style={{ fontSize: 10, fontWeight: 600, color: C.sub }}>{verdict}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Right: Soft Classification */}
                <div style={{ borderRadius: 10, border: `1px solid ${C.border}`, overflow: "hidden", display: "flex", flexDirection: "column" }}>
                    <div style={{ padding: "7px 13px", background: C.inputBg, borderBottom: `1px solid ${C.border}` }}>
                        <span style={{ fontSize: 9, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".06em" }}>Soft Classification</span>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", flex: 1 }}>
                        {[
                            { label: "Strong Fit", count: ev.bands["Strong Fit"], color: C.green, range: `≥ ${Math.round((ev.THRESH + 0.10) * 100)}%`, desc: "Top priority — interview now", tab: "decisions" },
                            { label: "Borderline", count: ev.bands["Borderline"], color: C.amber, range: `${Math.round((ev.THRESH - 0.05) * 100)}–${Math.round((ev.THRESH + 0.10) * 100)}%`, desc: "Review manually", tab: "decisions" },
                            { label: "Weak Fit", count: ev.bands["Weak Fit"], color: "#ef4444", range: `< ${Math.round((ev.THRESH - 0.05) * 100)}%`, desc: "Below threshold", tab: "decisions" },
                        ].map(({ label, count, color, range, desc }, i, arr) => (
                            <div key={label}
                                onClick={() => onNavDecisions && onNavDecisions(label)}
                                style={{
                                    flex: 1, padding: "0 16px",
                                    borderBottom: i < arr.length - 1 ? `1px solid ${C.border}` : "none",
                                    display: "flex", alignItems: "center", gap: 13,
                                    cursor: onNavDecisions ? "pointer" : "default",
                                    transition: "background .12s",
                                }}
                                onMouseEnter={e => { if (onNavDecisions) e.currentTarget.style.background = `${color}08`; }}
                                onMouseLeave={e => { e.currentTarget.style.background = "transparent"; }}
                            >
                                <div style={{ width: 4, alignSelf: "stretch", background: color, flexShrink: 0 }} />
                                <div style={{ flex: 1 }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                        <span style={{ fontSize: 24, fontWeight: 900, color, fontVariantNumeric: "tabular-nums", lineHeight: 1, minWidth: 28 }}>{count}</span>
                                        <div>
                                            <div style={{ fontSize: 12, fontWeight: 700, color }}>{label}</div>
                                            <div style={{ fontSize: 10, color: C.sub, marginTop: 2 }}>{range} · {desc}</div>
                                        </div>
                                    </div>
                                </div>
                                {onNavDecisions && <div style={{ fontSize: 9, color: `${color}99`, fontWeight: 700 }}>→</div>}
                            </div>
                        ))}
                    </div>
                </div>

            </div>
        </FieldCard>
    );
};


/**
 * AnalyticsView — deep analytics across three tabs:
 *   • Overview   — score distribution, bands, top-10 chart, talent quadrant
 *   • Decisions  — hiring shortlist, borderline review, rejection log with export
 *   • Skills     — skill coverage, gap analysis, top-3 radar comparison
 * Also renders ModelEvalMetrics (precision/recall/F1 approximations).
 *
 * Props:
 *   results     {Array}    Serialised candidate objects from the ML pipeline
 *   isMobile    {boolean}  Responsive layout switch (< 768 px)
 *   onNav       {function} Navigate to another tab
 *   activeModel {string}   Currently selected embedding model key
 */
const AnalyticsView = ({ results, isMobile, onNav, activeModel }) => {
    const [tab, setTab] = useState("overview");
    const [activeBand, setActiveBand] = useState(null);

    // Scroll main content area to top whenever the sub-tab changes
    useEffect(() => {
        document.querySelector("main")?.scrollTo({ top: 0, behavior: "smooth" });
    }, [tab]);

    if (!results || results.length === 0)
        return <EmptyState icon={BarChart2} title="No Analytics Yet" sub="Run the screening pipeline to generate analytics." />;

    const eligible = results.filter(c => c.eligible);
    const rejected = results.filter(c => !c.eligible);
    const top3 = eligible.slice(0, 3);
    const borderline = results.filter(c => c.band === "Borderline"); // Use backend band — avoids hardcoded threshold mismatch

    const avg = (arr, key) => arr.length ? Math.round(arr.reduce((a, c) => a + (c[key] || 0), 0) / arr.length * 100) : 0;
    const avgFinal = avg(eligible, "finalScore");
    const avgSkill = avg(eligible, "skillScore");
    const avgSem = avg(eligible, "semanticScore");
    const passRate = results.length ? Math.round(eligible.length / results.length * 100) : 0;
    const topScore = eligible.length ? Math.round((eligible[0].finalScore || 0) * 100) : 0;

    const skillCount = {}, missingCount = {};
    results.forEach(c => {
        (c.matched_skills || []).forEach(s => { skillCount[s] = (skillCount[s] || 0) + 1; });
        (c.missing_skills || []).forEach(s => { missingCount[s] = (missingCount[s] || 0) + 1; });
    });
    const coverageData = Object.keys({ ...skillCount, ...missingCount }).map(sk => ({
        skill: sk,
        matched: skillCount[sk] || 0,
        pct: Math.round(((skillCount[sk] || 0) / results.length) * 100),
    })).sort((a, b) => b.pct - a.pct);

    const gapData = Object.entries(missingCount).map(([skill, count]) => ({
        skill, count, pct: Math.round(count / results.length * 100)
    })).sort((a, b) => b.count - a.count);

    const buckets = { "0–20": 0, "21–40": 0, "41–60": 0, "61–80": 0, "81–100": 0 };
    results.forEach(c => {
        const s = Math.round((c.finalScore || 0) * 100);
        if (s <= 20) buckets["0–20"]++;
        else if (s <= 40) buckets["21–40"]++;
        else if (s <= 60) buckets["41–60"]++;
        else if (s <= 80) buckets["61–80"]++;
        else buckets["81–100"]++;
    });

    const quads = { tr: [], br: [], tl: [], bl: [] };
    results.forEach(c => {
        const sk = c.skillScore || 0, se = c.semanticScore || 0;
        if (sk >= 0.5 && se >= 0.45) quads.tr.push(c);
        else if (sk >= 0.5) quads.br.push(c);
        else if (se >= 0.45) quads.tl.push(c);
        else quads.bl.push(c);
    });

    const TABS = [
        { id: "overview", label: "Overview" },
        { id: "funnel", label: "Funnel" },
        { id: "talent", label: "Talent" },
        { id: "skills", label: "Skills" },
        { id: "decisions", label: "Decisions" },
    ];


    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>

            {/* Header */}
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
                <div>
                    <div style={{ fontSize: 20, fontWeight: 900, color: C.text, letterSpacing: "-.02em" }}>Recruitment Analytics</div>
                    <div style={{ fontSize: 11, color: C.sub, marginTop: 3 }}>
                        {results.length} screened · {eligible.length} shortlisted · {rejected.length} rejected
                    </div>
                </div>
                <LiquidTabBar sections={TABS} active={tab} onChange={setTab} />
            </div>

            {/* ═══ OVERVIEW ═══════════════════════════════════════════════════ */}
            {tab === "overview" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

                    {/* Compact KPI row */}
                    <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr 1fr" : "repeat(5,1fr)", gap: 10 }}>
                        {[
                            { label: "Screened", value: results.length, color: C.blue, sub: "total resumes" },
                            { label: "Shortlisted", value: eligible.length, color: C.green, sub: "passed filters" },
                            { label: "Rejected", value: rejected.length, color: "#ef4444", sub: "did not qualify" },
                            { label: "Pass Rate", value: `${passRate}%`, color: C.teal, sub: "of total pool" },
                            { label: "Top Score", value: `${topScore}%`, color: C.amber, sub: eligible[0]?.name?.split(" ")[0] || "—" },
                        ].map(p => <StatPill key={p.label} {...p} />)}
                    </div>

                    {/* Score distribution — glass bars, full names, no # prefix */}
                    <FieldCard label="Score Distribution" dot={C.blue} xtra={{ padding: "30px 14px 14px" }}>
                        <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                            {[...results].sort((a, b) => (b.finalScore || 0) - (a.finalScore || 0)).map((c, i) => {
                                const fs = Math.round((c.finalScore || 0) * 100);
                                const sk = Math.round((c.skillScore || 0) * 100);
                                const se = Math.round((c.semanticScore || 0) * 100);
                                const col = fs >= 70 ? C.green : fs >= 50 ? C.blue : C.amber;
                                const medals = ["🥇", "🥈", "🥉"];
                                return (
                                    <div key={c.id}
                                        title={`${c.name} — Final: ${fs}% · Skill: ${sk}% · Semantic: ${se}%`}
                                        style={{ display: "flex", alignItems: "center", gap: 10, borderRadius: 7, padding: "2px 4px", transition: "background .12s" }}
                                        onMouseEnter={e => e.currentTarget.style.background = C.tableFocus}
                                        onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                                        {/* Rank — medal or number */}
                                        <div style={{ width: 22, textAlign: "center", flexShrink: 0, fontSize: i < 3 ? 13 : 10, fontWeight: 700, color: C.muted, lineHeight: 1 }}>
                                            {i < 3 ? medals[i] : i + 1}
                                        </div>
                                        {/* Full name */}
                                        <div style={{ width: isMobile ? 80 : 130, fontSize: 11, fontWeight: 500, color: C.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flexShrink: 0 }}>
                                            {c.name || "Unknown"}
                                        </div>
                                        {/* Liquid glass bar — frosted track, saturated fill, inner sheen only on fill */}
                                        <div style={{ flex: 1, height: 20, borderRadius: 7, overflow: "hidden", background: C.inputBg, border: `1px solid ${C.border}`, backdropFilter: "blur(6px)", WebkitBackdropFilter: "blur(6px)", position: "relative" }}>
                                            {/* Fill */}
                                            <div style={{ position: "absolute", top: 0, left: 0, bottom: 0, width: `${Math.max(fs, 1)}%`, background: `linear-gradient(90deg,${col}88,${col}dd)`, transition: "width .85s cubic-bezier(.4,0,.2,1)", borderRadius: "7px 0 0 7px" }}>
                                                {/* Top sheen — only on fill area */}
                                                <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "42%", background: "linear-gradient(180deg,rgba(255,255,255,0.18),rgba(255,255,255,0))", borderRadius: "7px 0 0 0" }} />
                                                {/* Bottom inner shadow on fill */}
                                                <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: "30%", background: "rgba(0,0,0,0.15)", borderRadius: "0 0 0 7px" }} />
                                            </div>
                                            {/* Text overlay */}
                                            <div style={{ position: "relative", zIndex: 1, height: "100%", display: "flex", alignItems: "center", paddingLeft: 10, gap: 6 }}>
                                                <span style={{ fontSize: 10, fontWeight: 800, color: C.text, textShadow: "none", fontVariantNumeric: "tabular-nums" }}>{fs}%</span>

                                            </div>
                                        </div>
                                        {/* Status dot */}
                                        <div style={{ width: 6, height: 6, borderRadius: "50%", flexShrink: 0, background: c.eligible ? C.green : "#ef4444", opacity: .75 }} />
                                    </div>
                                );
                            })}
                        </div>
                        <div style={{ display: "flex", gap: 14, marginTop: 10, paddingTop: 8, borderTop: `1px solid ${C.border}` }}>
                            {[["● Shortlisted", "#22c55e"], ["● Rejected", "#ef4444"]].map(([l, c]) => (
                                <div key={l} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 9, color: C.muted }}>
                                    <span style={{ color: c }}>●</span>{l.slice(1)}
                                </div>
                            ))}
                        </div>
                    </FieldCard>

                    {/* ── Model Evaluation Metrics card ──────────────────── */}
                    <ModelEvalMetrics results={results} activeModel={activeModel} onNavDecisions={(band) => { setActiveBand(band || null); setTab("decisions"); }} />

                    {/* Score Bands — full width horizontal strip, matches score distribution width */}
                    <FieldCard label="Score Distribution Bands" dot={C.amber} xtra={{ padding: "24px" }}>

                        {/* ── Formula strip ─────────────────────────────────── */}
                        <div style={{ padding: "10px 12px", borderRadius: 8, background: C.inputBg, border: `1px solid ${C.border}`, marginBottom: 14 }}>
                            <div style={{ fontSize: 10, fontFamily: "monospace", color: C.text, lineHeight: 1, marginBottom: 8 }}>
                                Final = (<span style={{ color: C.blue }}>Skill Weight</span> × <span style={{ color: C.blue }}>Skill Score</span>) + (<span style={{ color: C.teal }}>Semantic Weight</span> × <span style={{ color: C.teal }}>Semantic Score</span>)
                            </div>
                            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                                <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.5 }}>
                                    <span style={{ color: C.blue, fontWeight: 700 }}>Skill Score</span> — % of required skills matched via keyword lookup, 150+ skill aliases, and semantic embedding similarity.
                                </div>
                                <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.5 }}>
                                    <span style={{ color: C.teal, fontWeight: 700 }}>Semantic Score</span> — cosine similarity of the resume embedding vs job description embedding. Measures meaning alignment beyond exact keywords.
                                </div>
                            </div>
                        </div>

                        {/* ── Bands table + avg stats ───────────────────────── */}
                        <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1px 160px", gap: 0, marginTop: 12, minWidth: 0, overflow: "hidden" }}>

                            {/* Left: distribution bands */}
                            <div style={{ paddingRight: isMobile ? 0 : 16, minWidth: 0 }}>
                                <div style={{ fontSize: 9, color: C.muted, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".07em", marginBottom: 10 }}>
                                    {results.length} resumes by final score
                                </div>
                                <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                                    {[
                                        { range: "81–100", label: "Excellent", desc: "strong fit", color: "#10b981" },
                                        { range: "61–80", label: "Good", desc: "worth interviewing", color: "#6395f1" },
                                        { range: "41–60", label: "Average", desc: "borderline", color: "#f3ac32" },
                                        { range: "21–40", label: "Low", desc: "weak match", color: "#e36b16" },
                                        { range: "0–20", label: "Very Low", desc: "poor fit", color: "#c11c1c" },
                                    ].map(({ range, label, desc, color }, idx, arr) => {
                                        const count = buckets[range] || 0;
                                        // Width proportional to total candidates — not relative to max-bucket
                                        const pct = results.length > 0 ? Math.round(count / results.length * 100) : 0;
                                        return (
                                            <div key={range} style={{ display: "grid", gridTemplateColumns: "80px 1fr 28px", alignItems: "center", gap: 8, padding: "6px 0", borderBottom: idx < arr.length - 1 ? `1px solid ${C.border}` : "none" }}>
                                                {/* Label */}
                                                <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                                                    <div style={{ width: 8, height: 8, borderRadius: 2, background: count ? color : `${color}30`, flexShrink: 0 }} />
                                                    <div>
                                                        <div style={{ fontSize: 10, fontWeight: 700, color: count ? C.text : C.muted, lineHeight: 1 }}>{label}</div>
                                                        <div style={{ fontSize: 8, color: C.muted, lineHeight: 1.2 }}>{range}</div>
                                                    </div>
                                                </div>
                                                {/* Bar */}
                                                <SoftBar pct={pct} color={color} h={6} />
                                                {/* Count */}
                                                <div style={{ fontSize: 12, fontWeight: 800, color: count ? color : C.muted, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{count}</div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>

                            {/* Divider */}
                            {!isMobile && <div style={{ background: C.border, margin: "0 0 0 0" }} />}

                            {/* Right: shortlisted avg */}
                            {!isMobile && (
                                <div style={{ paddingLeft: 16, paddingTop: 4, display: "flex", flexDirection: "column", justifyContent: "center", gap: 0, minWidth: 0 }}>
                                    <div style={{ fontSize: 9, color: C.muted, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".07em", marginBottom: 10 }}>
                                        Shortlisted avg
                                    </div>
                                    {[
                                        ["Skill", avgSkill, C.blue],
                                        ["Semantic", avgSem, C.teal],
                                        ["Final", avgFinal, C.amber],
                                    ].map(([l, v, col], i, arr) => (
                                        <div key={l} style={{ padding: "8px 0", borderBottom: i < arr.length - 1 ? `1px solid ${C.border}` : "none" }}>
                                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                                                <span style={{ fontSize: 10, fontWeight: 600, color: C.text }}>{l}</span>
                                                <span style={{ fontSize: 16, fontWeight: 900, color: col, fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>{v}%</span>
                                            </div>
                                            <div style={{ width: "100%", height: 4, borderRadius: 4, background: C.inputBg, overflow: "hidden" }}>
                                                <div style={{ height: "100%", width: `${v}%`, background: col, borderRadius: 4, transition: "width .9s cubic-bezier(.4,0,.2,1)" }} />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </FieldCard>

                    {/* Top Candidates — full width, 3-column card grid */}
                    {top3.length > 0 && (
                        <FieldCard label="Top Candidates" dot={C.amber} xtra={{ padding: "30px 14px 14px" }}>
                            <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "repeat(3,1fr)", gap: 12 }}>
                                {top3.map((c, i) => {
                                    const glow = [C.amber, "#94a3b8", "#cd7f32"][i];
                                    const medal = ["🥇", "🥈", "🥉"][i];
                                    const fs = Math.round((c.finalScore || 0) * 100);
                                    const sk = Math.round((c.skillScore || 0) * 100);
                                    const se = Math.round((c.semanticScore || 0) * 100);
                                    return (
                                        <div key={c.id} style={{ borderRadius: 14, background: `${glow}06`, border: `1px solid ${glow}20`, overflow: "hidden", position: "relative", display: "flex", flexDirection: "column" }}>
                                            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 1, background: `linear-gradient(90deg,transparent,${glow}55,transparent)` }} />
                                            {/* Header */}
                                            <div style={{ display: "flex", alignItems: "center", gap: 11, padding: "14px 14px 10px" }}>
                                                <div style={{ width: 36, height: 36, borderRadius: 9, background: `linear-gradient(135deg,${glow}33,${glow}18)`, border: `1px solid ${glow}30`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800, color: glow, flexShrink: 0 }}>
                                                    {(c.name || "?").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
                                                </div>
                                                <div style={{ flex: 1, minWidth: 0 }}>
                                                    <div style={{ fontSize: 12, fontWeight: 800, color: C.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.name}</div>
                                                    <div style={{ fontSize: 9, color: C.muted, marginTop: 1 }}>{medal} Rank #{i + 1}</div>
                                                </div>
                                                <div style={{ fontSize: 20, fontWeight: 900, color: glow, flexShrink: 0, fontVariantNumeric: "tabular-nums" }}>{fs}%</div>
                                            </div>
                                            {/* Score bar */}
                                            <div style={{ padding: "0 14px 10px" }}>
                                                <SoftBar pct={fs} color={glow} h={5} />
                                            </div>
                                            {/* Sk / Se chips */}
                                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, padding: "0 14px 12px" }}>
                                                {[["Skill", sk, C.blue], ["Semantic", se, C.teal]].map(([l, v, col]) => (
                                                    <div key={l} style={{ textAlign: "center", padding: "6px 8px", borderRadius: 8, background: `${col}08`, border: `1px solid ${col}16` }}>
                                                        <div style={{ fontSize: 13, fontWeight: 800, color: col }}>{v}%</div>
                                                        <div style={{ fontSize: 9, color: C.muted, marginTop: 1 }}>{l}</div>
                                                    </div>
                                                ))}
                                            </div>
                                            {/* Matched skills */}
                                            {(c.matched_skills || []).length > 0 && (
                                                <div style={{ padding: "0 14px 14px", display: "flex", flexWrap: "wrap", gap: 6, flex: 1 }}>
                                                    {(c.matched_skills || []).map(s => (
                                                        <span key={s} style={{ fontSize: 9, padding: "2px 8px", borderRadius: 20, background: `${glow}10`, color: glow, border: `1px solid ${glow}20`, fontWeight: 600 }}>{s}</span>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </FieldCard>
                    )}
                </div>
            )}

            {/* ═══ FUNNEL ══════════════════════════════════════════════════════ */}
            {tab === "funnel" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    <FieldCard label="Recruitment Funnel" dot={C.blue}>
                        <p style={{ fontSize: 12, color: C.cardText, marginBottom: 18, lineHeight: 1.6 }}>
                            Tracks how many candidates progressed through each ML pipeline stage — from raw PDF upload to final shortlist.
                        </p>
                        {(() => {
                            const stages = [
                                { label: "PDFs Uploaded", value: results.length, color: C.blue, desc: "Raw resumes received" },
                                { label: "Successfully Parsed", value: results.length, color: "#818cf8", desc: "Text extracted via pdfplumber / PyMuPDF" },
                                { label: "Profiles Extracted", value: results.length, color: C.teal, desc: "Name · email · skills · experience via regex + keyword rules" },
                                { label: "Scored & Ranked", value: results.length, color: C.amber, desc: `Avg final: ${avgFinal}% · skill: ${avgSkill}% · semantic: ${avgSem}%` },
                                { label: "Shortlisted", value: eligible.length, color: C.green, desc: `Pass rate: ${passRate}% of total pool` },
                            ];
                            return stages.map((s, i) => {
                                const pct = Math.round(s.value / (stages[0].value || 1) * 100);
                                const drop = i > 0 ? stages[i - 1].value - s.value : 0;
                                return (
                                    <div key={s.label}>
                                        {drop > 0 && (
                                            <div style={{ display: "flex", alignItems: "center", gap: 0, padding: "2px 0 2px 16px" }}>
                                                {/* Vertical connector aligned with the left border accent */}
                                                <div style={{ width: 2, height: 20, background: `rgba(239,68,68,0.3)`, borderRadius: 2, flexShrink: 0 }} />
                                                <div style={{ marginLeft: 12, display: "flex", alignItems: "center", gap: 5 }}>
                                                    <span style={{ fontSize: 9, color: "#ef4444", fontWeight: 700, letterSpacing: ".02em" }}>↓ {drop} candidate{drop > 1 ? "s" : ""} dropped</span>
                                                </div>
                                            </div>
                                        )}
                                        <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "14px 16px", borderRadius: 13, background: `${s.color}05`, border: `1px solid ${s.color}16`, marginBottom: 2, position: "relative", overflow: "hidden" }}>
                                            <div style={{ position: "absolute", top: 0, left: 0, bottom: 0, width: 2, background: s.color, opacity: .55, borderRadius: 2 }} />
                                            {/* Step number badge */}
                                            <div style={{ width: 26, height: 26, borderRadius: 8, background: `${s.color}18`, border: `1px solid ${s.color}30`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800, color: s.color, flexShrink: 0 }}>
                                                {i + 1}
                                            </div>
                                            <div style={{ flex: 1, minWidth: 0 }}>
                                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                                                    <div>
                                                        <span style={{ fontSize: 13, fontWeight: 700, color: C.text }}>{s.label}</span>
                                                        <span style={{ fontSize: 10, color: C.muted, marginLeft: 8 }}>{s.desc}</span>
                                                    </div>
                                                    <span style={{ fontSize: 15, fontWeight: 900, color: s.color, flexShrink: 0, fontVariantNumeric: "tabular-nums", marginLeft: 12 }}>{s.value}</span>
                                                </div>
                                                <SoftBar pct={pct} color={s.color} h={6} />
                                            </div>
                                        </div>
                                    </div>
                                );
                            });
                        })()}
                    </FieldCard>

                    <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr", gap: 14 }}>
                        {[
                            { label: "Shortlisted", count: eligible.length, color: C.green, list: eligible, icon: "✓" },
                            { label: "Rejected", count: rejected.length, color: "#ef4444", list: rejected, icon: "✗" },
                        ].map(({ label, count, color, list, icon }) => (
                            <FieldCard key={label} label={label} dot={color}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                                    <span style={{ fontSize: 13, fontWeight: 700, color }}>{count} candidates</span>
                                </div>
                                {/* Scrollable list — no "+N more" text */}
                                <div style={{ display: "flex", flexDirection: "column", gap: 5, maxHeight: 240, overflowY: "auto", paddingRight: 2 }}>
                                    {list.map(c => (
                                        <div key={c.id} style={{ display: "flex", alignItems: "center", gap: 9, padding: "7px 10px", borderRadius: 9, background: `${color}05`, border: `1px solid ${color}10`, flexShrink: 0 }}>
                                            <span style={{ fontSize: 10, color, fontWeight: 800, flexShrink: 0 }}>{icon}</span>
                                            <span style={{ fontSize: 12, color: C.text, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.name}</span>
                                            <span style={{ fontSize: 11, fontWeight: 700, color, flexShrink: 0, fontVariantNumeric: "tabular-nums" }}>{Math.round((c.finalScore || 0) * 100)}%</span>
                                        </div>
                                    ))}
                                </div>
                            </FieldCard>
                        ))}
                    </div>
                </div>
            )}

            {/* ═══ TALENT ══════════════════════════════════════════════════════ */}
            {tab === "talent" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

                    <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "repeat(3,1fr)", gap: 10 }}>
                        {[
                            { label: "Avg Skill Match", value: `${avgSkill}%`, color: C.blue, sub: "skill extraction + alias matching" },
                            { label: "Avg Semantic Score", value: `${avgSem}%`, color: C.teal, sub: "cosine similarity vs job description" },
                            { label: "Avg Final Score", value: `${avgFinal}%`, color: C.amber, sub: "weighted avg — shortlisted only" },
                        ].map(p => <StatPill key={p.label} {...p} />)}
                    </div>

                    <FieldCard label="Talent Quadrant Map" dot={C.teal}>
                        <p style={{ fontSize: 12, color: C.cardText, marginBottom: 14, lineHeight: 1.6 }}>
                            Candidates split by skill coverage vs semantic alignment. <strong style={{ color: C.text }}>Ideal Fit</strong> → strong on both → prioritise for interview.
                        </p>
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                            {[
                                { key: "tr", label: "Ideal Fit", sub: "High skill · High semantic", color: C.green, list: quads.tr, cta: "INTERVIEW NOW" },
                                { key: "br", label: "Skilled", sub: "High skill · Low semantic", color: C.blue, list: quads.br, cta: "CONSIDER" },
                                { key: "tl", label: "Role-Aware", sub: "Low skill · High semantic", color: C.amber, list: quads.tl, cta: "REVIEW" },
                                { key: "bl", label: "Weak Match", sub: "Low skill · Low semantic", color: "#ef4444", list: quads.bl, cta: "PASS" },
                            ].map(({ key, label, sub, color, list, cta }) => (
                                <div key={key} style={{ padding: "18px 18px 16px", borderRadius: 14, background: `${color}06`, border: `1px solid ${color}20`, position: "relative", overflow: "hidden" }}>
                                    <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 1, background: `linear-gradient(90deg,transparent,${color}55,transparent)` }} />
                                    {/* Header */}
                                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
                                        <div>
                                            <div style={{ fontSize: 14, fontWeight: 800, color, letterSpacing: "-.01em" }}>{label}</div>
                                            <div style={{ fontSize: 10, color: C.muted, marginTop: 2 }}>{sub}</div>
                                        </div>
                                        <div style={{ textAlign: "right" }}>
                                            <div style={{ fontSize: 26, fontWeight: 900, color, fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>{list.length}</div>
                                            <div style={{ fontSize: 8, fontWeight: 800, color, letterSpacing: ".06em", marginTop: 2, opacity: .8 }}>{cta}</div>
                                        </div>
                                    </div>
                                    {/* Candidate rows — scrollable, each with score */}
                                    <div style={{ display: "flex", flexDirection: "column", gap: 5, maxHeight: 160, overflowY: "auto", scrollbarWidth: "none" }}>
                                        {list.map(c => {
                                            const fs = Math.round((c.finalScore || 0) * 100);
                                            return (
                                                <div key={c.id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 9px", borderRadius: 8, background: `${color}08`, border: `1px solid ${color}14` }}>
                                                    <div style={{ width: 22, height: 22, borderRadius: 6, background: `${color}18`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 8, fontWeight: 800, color, flexShrink: 0 }}>
                                                        {(c.name || "?").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
                                                    </div>
                                                    <span style={{ fontSize: 11, color: C.text, fontWeight: 600, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.name}</span>
                                                    <span style={{ fontSize: 11, fontWeight: 800, color, flexShrink: 0, fontVariantNumeric: "tabular-nums" }}>{fs}%</span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </FieldCard>

                    <FieldCard label="Top 10 Talent Rankings" dot={C.blue}>
                        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
                            <button onClick={() => onNav && onNav("candidates")} style={{ fontSize: 11, fontWeight: 600, color: C.blue, background: `${C.blue}10`, border: `1px solid ${C.blue}25`, borderRadius: 8, padding: "4px 12px", cursor: "pointer", fontFamily: "inherit", display: "flex", alignItems: "center", gap: 5 }}>
                                View All Candidates →
                            </button>
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            {eligible.slice(0, 10).map((c, i) => {
                                const fs = Math.round((c.finalScore || 0) * 100);
                                const sk = Math.round((c.skillScore || 0) * 100);
                                const se = Math.round((c.semanticScore || 0) * 100);
                                const rankColor = i === 0 ? C.amber : i === 1 ? "#94a3b8" : i === 2 ? "#cd7f32" : C.blue;
                                const medals = ["🥇", "🥈", "🥉"];
                                return (
                                    <div key={c.id} style={{ display: "flex", alignItems: "center", gap: 12, padding: "11px 14px", borderRadius: 12, background: i < 3 ? `${rankColor}07` : C.surface, border: `1px solid ${i < 3 ? `${rankColor}20` : C.border}`, transition: "background .15s" }}
                                        onMouseEnter={e => e.currentTarget.style.background = `${C.blue}07`}
                                        onMouseLeave={e => e.currentTarget.style.background = i < 3 ? `${rankColor}07` : C.surface}>
                                        {/* Rank */}
                                        <div style={{ width: 28, textAlign: "center", flexShrink: 0, fontSize: i < 3 ? 15 : 11, fontWeight: 800, color: rankColor, lineHeight: 1 }}>
                                            {i < 3 ? medals[i] : i + 1}
                                        </div>
                                        {/* Avatar + name */}
                                        <div style={{ width: 30, height: 30, borderRadius: 8, background: `${rankColor}18`, border: `1px solid ${rankColor}28`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 800, color: rankColor, flexShrink: 0 }}>
                                            {(c.name || "?").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
                                        </div>
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <div style={{ fontSize: 12, fontWeight: 700, color: C.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.name}</div>
                                            <div style={{ fontSize: 9, color: C.muted }}>Skill {sk}% · Semantic {se}%</div>
                                        </div>
                                        {/* Talent score bar + final % */}
                                        <div style={{ width: 100, flexShrink: 0 }}>
                                            <SoftBar pct={fs} color={rankColor} h={5} />
                                        </div>
                                        <div style={{ width: 36, textAlign: "right", fontSize: 13, fontWeight: 900, color: rankColor, flexShrink: 0, fontVariantNumeric: "tabular-nums" }}>{fs}%</div>
                                    </div>
                                );
                            })}
                        </div>
                    </FieldCard>
                </div>
            )}

            {/* ═══ SKILLS ══════════════════════════════════════════════════════ */}
            {tab === "skills" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 12 }}>

                        <FieldCard label="Skill Coverage" dot={C.green} xtra={{ padding: "16px" }}>
                            <p style={{ fontSize: 11, color: C.muted, marginBottom: 12 }}>How many of the {results.length} screened candidates match each required skill.</p>
                            <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr", gap: "6px 20px" }}>
                                {coverageData.map(({ skill, matched, pct }) => {
                                    const col = pct >= 70 ? C.green : pct >= 40 ? C.blue : pct >= 20 ? C.amber : "#ef4444";
                                    return (
                                        <div key={skill}>
                                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                                                <span style={{ fontSize: 11, fontWeight: 600, color: C.text }}>{skill}</span>
                                                <span style={{ fontSize: 11, fontWeight: 700, color: col, fontVariantNumeric: "tabular-nums" }}>{matched}/{results.length} <span style={{ fontWeight: 400, color: C.muted }}>({pct}%)</span></span>
                                            </div>
                                            <SoftBar pct={pct} color={col} h={5} />
                                        </div>
                                    );
                                })}
                            </div>
                        </FieldCard>

                        <FieldCard label="Skill Gap Analysis" dot="#ef4444" xtra={{ border: `1px solid rgba(239,68,68,.2)`, padding: "16px" }}>
                            <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
                                {[
                                    { tag: "RARE", col: "#ef4444", desc: "70%+ missing — very hard to find" },
                                    { tag: "SCARCE", col: "#f97316", desc: "40–70% missing — hard to find" },
                                    { tag: "LOW", col: C.amber, desc: "20–40% missing — some lack this" },
                                    { tag: "OK", col: C.teal, desc: "Under 20% — most have this" },
                                ].map((item) => (
                                    <div key={item.tag} style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 10px", borderRadius: 8, background: `${item.col}08`, border: `1px solid ${item.col}22` }}>
                                        <span style={{ fontSize: 9, padding: "1px 6px", borderRadius: 4, fontWeight: 800, background: `${item.col}15`, color: item.col, border: `1px solid ${item.col}25`, fontFamily: "monospace" }}>{item.tag}</span>
                                        <span style={{ fontSize: 10, color: C.text }}>{item.desc}</span>
                                    </div>
                                ))}
                            </div>
                            <p style={{ fontSize: 11, color: C.muted, marginBottom: 14, lineHeight: 1.6 }}>
                                The bar shows what % of candidates are <strong style={{ color: C.text }}>missing</strong> each required skill. Longer red bar = harder skill to find in your talent pool = consider relaxing that requirement or sourcing from a different pool.
                            </p>
                            {gapData.length === 0 ? (
                                <div style={{ textAlign: "center", padding: "20px", color: C.green, fontSize: 13, fontWeight: 700, borderRadius: 10, background: `${C.green}07`, border: `1px solid ${C.green}20` }}>All required skills are well represented in this talent pool</div>
                            ) : (
                                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                    {gapData.map(({ skill, count, pct }) => {
                                        const col = pct >= 70 ? "#ef4444" : pct >= 40 ? "#f97316" : pct >= 20 ? C.amber : C.teal;
                                        const tag = pct >= 70 ? "RARE" : pct >= 40 ? "SCARCE" : pct >= 20 ? "LOW" : "OK";
                                        return (
                                            <div key={skill}>
                                                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                                                    <div style={{ flex: 1, fontSize: 12, fontWeight: 600, color: C.text }}>{skill}</div>
                                                    <span style={{ fontSize: 10, color: col, fontWeight: 700 }}>{count} of {results.length} candidates missing</span>
                                                    <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, fontWeight: 800, background: `${col}14`, color: col, border: `1px solid ${col}22`, fontFamily: "monospace", minWidth: 40, textAlign: "center" }}>{tag}</span>
                                                </div>
                                                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                                    <SoftBar pct={pct} color={col} h={7} />
                                                    <span style={{ fontSize: 11, fontWeight: 700, color: col, minWidth: 32, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{pct}%</span>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </FieldCard>
                    </div>

                    {eligible.length >= 1 && (
                        <FieldCard label="Top 3 Candidates — Skill Radar Comparison" dot={C.blue} xtra={{ padding: "24px 18px 20px" }}>
                            <p style={{ fontSize: 11, color: C.muted, marginBottom: 18, lineHeight: 1.6 }}>
                                Each axis = a required skill. Filled area = how well the candidate matches that skill. Missing skills appear near the centre. Hover any point for the score.
                            </p>
                            <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.min(eligible.length, 3)},1fr)`, gap: 16 }}>
                                {eligible.slice(0, 3).map((c0, ci) => {
                                    const colors = [C.blue, C.teal, "#a78bfa"];
                                    const col = colors[ci];
                                    const matchedSet = new Set((c0.matched_skills || []).map(s => s.toLowerCase()));
                                    // All required skills — no truncation; binary: matched = 100, missing = 0
                                    const allSkills = [...new Set([...(c0.matched_skills || []), ...(c0.missing_skills || [])])];
                                    const sk = Math.round((c0.skillScore || 0) * 100);
                                    const se = Math.round((c0.semanticScore || 0) * 100);
                                    const totalReq = allSkills.length;
                                    const matchedCnt = (c0.matched_skills || []).length;
                                    const radarD = allSkills.map(s => ({
                                        skill: s.length > 16 ? s.slice(0, 14) + "…" : s,
                                        fullName: s,           // untruncated — used by tooltip
                                        score: matchedSet.has(s.toLowerCase()) ? 100 : 0,
                                        fullMark: 100,
                                    }));
                                    return (
                                        <div key={c0.id} style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                                            <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 8, padding: "4px 12px", borderRadius: 20, background: `${col}10`, border: `1px solid ${col}22` }}>
                                                <div style={{ width: 7, height: 7, borderRadius: "50%", background: col }} />
                                                <span style={{ fontSize: 11, fontWeight: 700, color: col }}>#{ci + 1}</span>
                                                <span style={{ fontSize: 11, fontWeight: 600, color: C.text, maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c0.name}</span>
                                            </div>
                                            <div style={{ fontSize: 10, color: C.muted, marginBottom: 6, textAlign: "center" }}>
                                                Skill: {sk}% · Semantic: {se}% · Final: {Math.round((c0.finalScore || 0) * 100)}%
                                            </div>
                                            <ResponsiveContainer width="100%" height={Math.max(220, allSkills.length * 20)}>
                                                <RadarChart data={radarD} outerRadius="68%" margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                                                    <PolarGrid gridType="circle" stroke={C.border} />
                                                    <PolarAngleAxis dataKey="skill" tick={{ fill: C.text, fontSize: 9, fontWeight: 600 }} />
                                                    <Tooltip
                                                        content={({ payload }) => {
                                                            if (!payload?.length) return null;
                                                            const { fullName, score } = payload[0].payload;
                                                            const isMatched = score === 100;
                                                            const coveragePct = Math.round(100 / totalReq);
                                                            return (
                                                                <div style={{ background: "rgba(18,18,22,0.97)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", border: `1px solid ${isMatched ? col : "#ef4444"}`, borderRadius: 10, padding: "9px 13px", fontSize: 11, color: "#f1f1f3", minWidth: 170, boxShadow: "0 4px 16px rgba(0,0,0,.45)" }}>
                                                                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 5 }}>
                                                                        <div style={{ width: 8, height: 8, borderRadius: "50%", background: isMatched ? col : "#ef4444", flexShrink: 0 }} />
                                                                        <span style={{ fontWeight: 800, color: isMatched ? col : "#ef4444", fontSize: 11 }}>{isMatched ? "Matched" : "Missing"}</span>
                                                                    </div>
                                                                    <div style={{ fontWeight: 700, color: "#f1f1f3", marginBottom: 6, lineHeight: 1.3 }}>{fullName}</div>
                                                                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, padding: "5px 0", borderTop: `1px solid rgba(255,255,255,0.10)` }}>
                                                                        <span style={{ color: "#9ca3af", fontSize: 10 }}>This skill</span>
                                                                        <span style={{ fontWeight: 700, color: "#f1f1f3", fontVariantNumeric: "tabular-nums" }}>1 of {totalReq} required</span>
                                                                    </div>
                                                                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
                                                                        <span style={{ color: "#9ca3af", fontSize: 10 }}>Overall skills</span>
                                                                        <span style={{ fontWeight: 700, color: isMatched ? col : "#ef4444", fontVariantNumeric: "tabular-nums" }}>{matchedCnt}/{totalReq} matched</span>
                                                                    </div>
                                                                </div>
                                                            );
                                                        }}
                                                    />
                                                    <Radar dataKey="score" stroke={col} strokeWidth={2} fill={col} fillOpacity={0.18} isAnimationActive={false} dot={{ fill: col, r: 4, strokeWidth: 1.5, stroke: "rgba(255,255,255,0.2)" }} activeDot={{ r: 6, fill: col, stroke: "#fff", strokeWidth: 2 }} />
                                                </RadarChart>
                                            </ResponsiveContainer>
                                        </div>
                                    );
                                })}
                            </div>
                        </FieldCard>
                    )}
                </div>
            )}

            {/* ═══ DECISIONS ═══════════════════════════════════════════════════ */}
            {tab === "decisions" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

                    {/* Hiring Recommendation — full priority layout like quadrant map */}
                    <FieldCard label="Hiring Recommendation" dot={C.green}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14, flexWrap: "wrap", gap: 8 }}>
                            <p style={{ fontSize: 11, color: C.sub, lineHeight: 1.6, margin: 0 }}>
                                {activeBand
                                    ? <>Filtered by: <strong style={{ color: activeBand === "Strong Fit" ? C.green : activeBand === "Borderline" ? C.amber : "#ef4444" }}>{activeBand}</strong></>
                                    : "Action-prioritised summary for the HR team. Review each group and proceed accordingly."
                                }
                            </p>
                            {activeBand && (
                                <button onClick={() => setActiveBand(null)} style={{ fontSize: 10, fontWeight: 700, color: C.sub, background: C.inputBg, border: `1px solid ${C.border}`, borderRadius: 8, padding: "3px 10px", cursor: "pointer", fontFamily: "inherit", whiteSpace: "nowrap" }}>
                                    ✕ Show All
                                </button>
                            )}
                        </div>
                        {(() => {
                            // Single source of truth: backend band field
                            const interviewNow = results.filter(c => c.band === "Strong Fit");
                            const reviewList = results.filter(c => c.band === "Borderline");
                            const archiveList = results.filter(c => !c.eligible);
                            const allGroups = [
                                { label: "Interview Now", list: interviewNow, band: "Strong Fit", color: C.green, action: "Strong match — schedule interview", priority: "HIGH PRIORITY", exportList: interviewNow },
                                { label: "Secondary Review", list: reviewList, band: "Borderline", color: C.amber, action: "Close to threshold — manual review recommended", priority: "REVIEW", exportList: reviewList },
                                { label: "Archive", list: archiveList, band: "Weak Fit", color: "#ef4444", action: "Did not meet criteria — no further action required", priority: "ARCHIVE", exportList: null },
                            ];
                            // If a band was selected from the soft classification, show only that group
                            const groups = activeBand
                                ? allGroups.filter(g => g.band === activeBand)
                                : allGroups;
                            return (
                                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                    {groups.map(({ label, list, color, action, priority, exportList }) => (
                                        <div key={label} style={{ borderRadius: 14, border: `1px solid ${color}20`, overflow: "hidden", background: `${color}04` }}>
                                            {/* Group header */}
                                            <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", borderBottom: list.length ? `1px solid ${color}12` : "none", background: `${color}07` }}>
                                                <div style={{ position: "relative" }}>
                                                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: color }} />
                                                    <div style={{ position: "absolute", inset: 0, borderRadius: "50%", background: color, opacity: .35, transform: "scale(2.2)" }} />
                                                </div>
                                                <div style={{ flex: 1 }}>
                                                    <div style={{ fontSize: 13, fontWeight: 800, color }}>{label}</div>
                                                    <div style={{ fontSize: 10, color: C.muted, marginTop: 1 }}>{action}</div>
                                                </div>
                                                <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                                                    {exportList && exportList.length > 0 && (
                                                        <button onClick={() => exportDecisionGroup(exportList, label)} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 9, padding: "3px 9px", borderRadius: 20, fontWeight: 700, background: `${color}14`, color, border: `1px solid ${color}35`, cursor: "pointer", fontFamily: "inherit" }}>
                                                            <Download size={9} /> Export ({exportList.length})
                                                        </button>
                                                    )}
                                                    <span style={{ fontSize: 9, padding: "3px 9px", borderRadius: 20, fontWeight: 800, background: `${color}14`, color, border: `1px solid ${color}28`, letterSpacing: ".05em" }}>{priority}</span>
                                                    <span style={{ fontSize: 20, fontWeight: 900, color, fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>{list.length}</span>
                                                </div>
                                            </div>
                                            {/* Scrollable candidate rows */}
                                            {list.length > 0 && (
                                                <div style={{ display: "flex", flexDirection: "column", gap: 0, maxHeight: 180, overflowY: "auto", scrollbarWidth: "none" }}>
                                                    {list.map((c, idx) => {
                                                        const fs = Math.round((c.finalScore || 0) * 100);
                                                        const sk = Math.round((c.skillScore || 0) * 100);
                                                        const se = Math.round((c.semanticScore || 0) * 100);
                                                        return (
                                                            <div key={c.id} style={{ display: "flex", alignItems: "center", gap: 12, padding: "9px 16px", borderBottom: idx < list.length - 1 ? `1px solid ${color}08` : "none", transition: "background .12s" }}
                                                                onMouseEnter={e => e.currentTarget.style.background = `${color}07`}
                                                                onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                                                                <div style={{ width: 26, height: 26, borderRadius: 7, background: `${color}14`, border: `1px solid ${color}22`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 8, fontWeight: 800, color, flexShrink: 0 }}>
                                                                    {(c.name || "?").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
                                                                </div>
                                                                <div style={{ flex: 1, minWidth: 0 }}>
                                                                    <div style={{ fontSize: 12, fontWeight: 600, color: C.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.name}</div>
                                                                    <div style={{ fontSize: 9, color: C.muted, marginTop: 1 }}>Skill: {sk}% · Semantic: {se}%</div>
                                                                </div>
                                                                <div style={{ width: 70, flexShrink: 0 }}>
                                                                    <SoftBar pct={fs} color={color} h={4} />
                                                                </div>
                                                                <span style={{ fontSize: 12, fontWeight: 800, color, flexShrink: 0, fontVariantNumeric: "tabular-nums", minWidth: 32, textAlign: "right" }}>{fs}%</span>
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            )}
                                            {list.length === 0 && (
                                                <div style={{ padding: "12px 16px", fontSize: 11, color: C.muted }}>None in this group.</div>
                                            )}

                                        </div>
                                    ))}
                                </div>
                            );
                        })()}
                    </FieldCard>

                    {/* Borderline */}
                    {borderline.length > 0 && (!activeBand || activeBand === "Borderline") && (
                        <div id="borderline-section" style={{ scrollMarginTop: 16 }}><FieldCard label="Borderline Review" dot={C.amber} xtra={{ border: `1px solid ${C.amber}22` }}>
                            <p style={{ fontSize: 12, color: C.cardText, marginBottom: 14, lineHeight: 1.6 }}>
                                These candidates are <strong style={{ color: C.amber }}>close to the screening threshold</strong> — shortlisted but near the cutoff. Recommend manual review before final decision.
                            </p>
                            <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                                {borderline.map(c => {
                                    const fs = Math.round((c.finalScore || 0) * 100), sk = Math.round((c.skillScore || 0) * 100), se = Math.round((c.semanticScore || 0) * 100);
                                    return (
                                        <div key={c.id} style={{ display: "flex", alignItems: "center", gap: 12, padding: "11px 14px", borderRadius: 12, background: `${C.amber}05`, border: `1px solid ${C.amber}15` }}>
                                            <div style={{ width: 30, height: 30, borderRadius: 8, background: `${C.amber}15`, border: `1px solid ${C.amber}25`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 800, color: C.amber, flexShrink: 0 }}>
                                                {(c.name || "?").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
                                            </div>
                                            <div style={{ flex: 1, minWidth: 0 }}>
                                                <div style={{ fontSize: 12, fontWeight: 700, color: C.text }}>{c.name}</div>
                                                <div style={{ fontSize: 10, color: C.muted }}>{c.email || "No email"}</div>
                                            </div>
                                            <div style={{ display: "flex", gap: 5, flexShrink: 0 }}>
                                                {[["Skill", sk, C.blue], ["Semantic", se, C.teal], ["Score", fs, C.amber]].map(([l, v, col]) => (
                                                    <div key={l} style={{ textAlign: "center", padding: "4px 7px", borderRadius: 7, background: `${col}09`, border: `1px solid ${col}18`, minWidth: 36 }}>
                                                        <div style={{ fontSize: 11, fontWeight: 800, color: col, fontVariantNumeric: "tabular-nums" }}>{v}%</div>
                                                        <div style={{ fontSize: 8, color: C.muted }}>{l}</div>
                                                    </div>
                                                ))}
                                            </div>
                                            <span style={{ fontSize: 9, padding: "3px 8px", borderRadius: 20, background: `${C.amber}14`, color: C.amber, border: `1px solid ${C.amber}30`, fontWeight: 700, fontFamily: "monospace", flexShrink: 0 }}>REVIEW</span>
                                        </div>
                                    );
                                })}
                            </div>
                        </FieldCard></div>
                    )}

                    {/* Rejection log */}
                    {rejected.length > 0 && (!activeBand || activeBand === "Weak Fit") && (
                        <FieldCard label={`Rejection Log — ${rejected.length} candidate${rejected.length !== 1 ? "s" : ""}`} dot="#ef4444" xtra={{ border: "1px solid rgba(239,68,68,.18)" }}>
                            <p style={{ fontSize: 11, color: C.muted, marginBottom: 12, lineHeight: 1.6 }}>
                                Each candidate below was disqualified during the eligibility filter stage — before scoring. The reason and a brief analysis is shown for each.
                            </p>
                            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                {rejected.map((c, i) => {
                                    const sk = Math.round((c.skillScore || 0) * 100);
                                    const se = Math.round((c.semanticScore || 0) * 100);
                                    const fs = Math.round((c.finalScore || 0) * 100);
                                    const thresh = Math.round((c.dynamic_threshold || 0.5) * 100);
                                    const reason = c.rejection_reason || "Did not meet criteria";
                                    const code = c.rejection_code || "";
                                    const isDomainMismatch = se < 35;
                                    let strengths = [], gaps = [], recommendation = "";
                                    if (code === "no_contact") {
                                        strengths = sk > 40 ? [`Skill coverage: ${sk}%`] : se > 45 ? [`Role alignment: ${se}%`] : [];
                                        gaps = ["No contact information found in the resume (email or phone missing)"];
                                        recommendation = `Cannot proceed to interview stage without contact details. Ask candidate to resubmit with valid email or phone number.`;

                                    } else if (code === "skill_below_min") {
                                        strengths = se > 50 ? [`Role alignment: ${se}%`] : [];
                                        if (isDomainMismatch) {
                                            gaps = [
                                                `Only ${sk}% of required skills matched (minimum 30%)`,
                                                `Very low semantic alignment (${se}%) — indicates a likely domain mismatch`,
                                            ];
                                            recommendation = "Candidate's background appears to be from a different field (e.g. aeronautical, civil, mechanical). This role requires a different skill domain. Not suitable unless cross-domain skills are acceptable.";
                                        } else {
                                            gaps = [`Only ${sk}% of required skills matched — minimum threshold is 30%`];
                                            recommendation = `Candidate lacks core skills for this role. Suitable for junior positions or roles with relaxed skill requirements.`;
                                        }

                                    } else if (code === "experience_below_min") {
                                        strengths = sk > 50 ? [`Skill coverage: ${sk}%`] : [];
                                        gaps = [reason || "Experience below the configured minimum requirement"];
                                        recommendation = "Does not meet the seniority level required. Consider for a junior or entry-level variant of this role if skill alignment is strong.";

                                    } else if (code === "missing_education") {
                                        strengths = sk > 50 ? [`Skill coverage: ${sk}%`] : se > 50 ? [`Role alignment: ${se}%`] : [];
                                        gaps = ["Required educational qualification was not detected in the resume"];
                                        recommendation = "Verify if the qualification exists under a different format or abbreviation. If the education requirement is flexible, consider manual review.";

                                    } else if (code === "score_below_threshold") {
                                        if (isDomainMismatch) {
                                            strengths = sk > 50 ? [`Skill coverage: ${sk}%`] : [];
                                            gaps = [
                                                `Final score ${fs}% is below the batch threshold of ${thresh}%`,
                                                `Semantic alignment is very low (${se}%) — likely a domain mismatch`,
                                            ];
                                            recommendation = "Candidate's background appears unrelated to this role's domain. In a competitive batch, this candidate does not reach the shortlist threshold. No further action recommended.";
                                        } else {
                                            strengths = sk > 50 ? [`Skill coverage: ${sk}%`] : se > 50 ? [`Role alignment: ${se}%`] : [];
                                            gaps = [`Final score ${fs}% is below the competitive batch threshold of ${thresh}%`];
                                            recommendation = `Below the cutoff for this specific batch. This candidate may be suitable in a less competitive pool or for a different role.`;
                                        }

                                    } else {
                                        strengths = sk > 50 ? [`Skill coverage: ${sk}%`] : [];
                                        if (isDomainMismatch) {
                                            gaps = [
                                                `Semantic alignment ${se}% — likely unrelated field`,
                                                `Final score: ${fs}%`,
                                            ];
                                            recommendation = "Candidate's background appears to be outside the required domain. Consider only if cross-domain experience is acceptable.";
                                        } else {
                                            gaps = [`Role alignment: ${se}%`, `Final score: ${fs}%`];
                                            recommendation = "Candidate does not meet key requirements for this batch. Manual review warranted only with strong external factors (e.g. referral).";
                                        }
                                    }
                                    return (
                                        <div key={c.id || i} style={{ borderRadius: 12, border: "1px solid rgba(239,68,68,.14)", overflow: "hidden" }}>
                                            {/* Header row */}
                                            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", background: "rgba(239,68,68,.05)", flexWrap: "wrap" }}>
                                                <div style={{ width: 32, height: 32, borderRadius: 8, background: "rgba(239,68,68,.12)", border: "1px solid rgba(239,68,68,.2)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 800, color: "#ef4444", flexShrink: 0 }}>
                                                    {(c.name || "?").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
                                                </div>
                                                <div style={{ flex: 1, minWidth: 120 }}>
                                                    <div style={{ fontSize: 13, fontWeight: 700, color: C.text }}>{c.name || "Unknown"}</div>
                                                    <div style={{ fontSize: 10, color: "#ef4444", marginTop: 1, fontWeight: 600 }}>✗ {reason}</div>
                                                    {isDomainMismatch && (
                                                        <div style={{ fontSize: 10, color: "#f97316", marginTop: 2, fontWeight: 600 }}>⚠ Domain mismatch likely — low semantic alignment ({se}%)</div>
                                                    )}
                                                </div>
                                                <div style={{ display: "flex", gap: 6, flexShrink: 0, flexWrap: "wrap" }}>
                                                    {[["Skill", sk, C.blue], ["Semantic", se, C.teal], ["Final", fs, "#ef4444"]].map(([l, v, col]) => (
                                                        <div key={l} style={{ textAlign: "center", padding: "4px 8px", borderRadius: 7, background: `${col}09`, border: `1px solid ${col}18`, minWidth: 44 }}>
                                                            <div style={{ fontSize: 12, fontWeight: 800, color: col, fontVariantNumeric: "tabular-nums" }}>{v}%</div>
                                                            <div style={{ fontSize: 8, color: C.muted }}>{l}</div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                            {/* Structured rejection analysis */}
                                            <div style={{ padding: "10px 14px 12px", background: "rgba(239,68,68,.02)", borderTop: "1px solid rgba(239,68,68,.08)", display: "flex", flexDirection: "column", gap: 5 }}>
                                                {strengths.length > 0 && (
                                                    <div style={{ display: "flex", gap: 8 }}>
                                                        <span style={{ fontSize: 9, fontWeight: 700, color: C.teal, textTransform: "uppercase", letterSpacing: ".05em", minWidth: 68, paddingTop: 2, flexShrink: 0 }}>Strengths : </span>
                                                        <span style={{ fontSize: 11, color: C.cardText, lineHeight: 1.5 }}>{strengths.join(" · ")}</span>
                                                    </div>
                                                )}
                                                <div style={{ display: "flex", gap: 8 }}>
                                                    <span style={{ fontSize: 9, fontWeight: 700, color: "#ef4444", textTransform: "uppercase", letterSpacing: ".05em", minWidth: 68, paddingTop: 2, flexShrink: 0 }}>Gaps : </span>
                                                    <span style={{ fontSize: 11, color: C.cardText, lineHeight: 1.5 }}>{gaps.join(" · ")}</span>
                                                </div>
                                                <div style={{ display: "flex", gap: 8 }}>
                                                    <span style={{ fontSize: 9, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".05em", minWidth: 68, paddingTop: 2, flexShrink: 0 }}>Action : </span>
                                                    <span style={{ fontSize: 11, color: C.text, lineHeight: 1.5, fontWeight: 500 }}>{recommendation}</span>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </FieldCard>
                    )}
                </div>
            )}

        </div>
    );
};

// Defined outside App() to prevent remounting on every parent render
/**
 * SidebarContent — rendered inside both the persistent desktop sidebar and the
 * mobile slide-in drawer. Shows the nav items, backend status indicator, and
 * a live results count badge when candidates are available.
 *
 * Props:
 *   navItems      {Array}    Navigation item definitions
 *   nav           {string}   Currently active tab id
 *   go            {function} Navigate to a tab
 *   backendOnline {boolean|null} null = checking, true = online, false = offline
 *   results       {Array}    Used to show a candidate count badge on the sidebar
 */
const SidebarContent = ({ navItems, nav, go, backendOnline, results }) => (
    <>
        {/* Logo */}
        <div style={{ padding: "0 8px 20px", borderBottom: `1px solid ${C.border}`, marginBottom: 11 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ width: 36, height: 36, borderRadius: 9, background: `linear-gradient(135deg,${C.blue},#a78bfa)`, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: `0 4px 12px ${C.blue}44`, flexShrink: 0 }}>
                    <Sparkles size={15} color="#fff" />
                </div>
                <div>
                    <div style={{ fontSize: 13, fontWeight: 800, color: C.text, lineHeight: 1.1 }}>ML-Based</div>
                    <div style={{ fontSize: 10, color: C.sub, marginTop: 1 }}>Screening System</div>
                </div>
            </div>
            {/* Developer credit */}
            <div style={{ marginTop: 20, padding: "7px 9px", borderRadius: 8, background: `${C.blue}08`, border: `1px solid ${C.blue}18` }}>
                <div style={{ fontSize: 9, color: C.muted, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 2 }}>Developed by</div>
                <div style={{ fontSize: 12, fontWeight: 800, color: C.blue }}>Madhan Kumar</div>
                <div style={{ fontSize: 9, color: C.sub }}>B.Sc CS · Final Year Project</div>
            </div>
        </div>

        {/* Nav items */}
        <nav style={{ flex: 1, display: "flex", flexDirection: "column", gap: 2 }}>
            {navItems.map(({ id, icon: Icon, label }) => {
                const active = nav === id;
                return (
                    <button key={id} onClick={() => go(id)} style={{
                        width: "100%", display: "flex", alignItems: "center", gap: 10,
                        padding: "9px 10px", borderRadius: 9, border: "none",
                        background: active ? `${C.blue}14` : "transparent",
                        color: active ? C.blue : C.sub,
                        cursor: "pointer", fontSize: 13, fontWeight: active ? 700 : 500,
                        transition: "all .13s", fontFamily: "inherit",
                        borderLeft: `2px solid ${active ? C.blue : "transparent"}`,
                    }}
                        onMouseEnter={e => { if (!active) e.currentTarget.style.color = C.text; }}
                        onMouseLeave={e => { if (!active) e.currentTarget.style.color = C.sub; }}
                    >
                        <Icon size={14} /> {label}
                    </button>
                );
            })}
        </nav>

        {/* Pipeline status pill — reflects /api/health poll */}
        {(() => {
            const isOff = backendOnline === false;
            const isChecking = backendOnline === null;
            const col = isOff ? "#ef4444" : isChecking ? C.muted : C.green;
            const bg = isOff ? "rgba(239,68,68,.08)" : isChecking ? `${C.muted}08` : `${C.green}08`;
            const bd = isOff ? "rgba(239,68,68,.20)" : isChecking ? `${C.muted}20` : `${C.green}20`;
            const lbl = isOff ? "Pipeline Offline"
                : isChecking ? "Checking\u2026"
                    : results.length > 0 ? `${results.length} candidates ranked`
                        : "Pipeline Ready";
            return (
                <div style={{ padding: "8px 10px", borderRadius: 8, background: bg, border: `1px solid ${bd}`, fontSize: 11, color: col, display: "flex", alignItems: "center", gap: 7 }}>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: col, flexShrink: 0, animation: (!isOff && !isChecking) ? "pulse 2s infinite" : "none" }} />
                    {lbl}
                </div>
            );
        })()}
    </>
);

/**
 * App — root component. Owns all shared state:
 *   • dark / light theme toggle
 *   • active navigation tab
 *   • screeningConfig  — parameters passed from UploadView to ProcessingView
 *   • results          — serialised candidate output from the ML pipeline
 *   • activeModel      — selected embedding model key (mpnet / mxbai / arctic)
 *   • activeSessionRef — ref tracking the backend session_id for disk cleanup
 *
 * Session lifecycle:
 *   1. UploadView calls handleStartScreening → clears any previous session.
 *   2. ProcessingView uploads files, calls onSessionReady(session_id) → stored in ref.
 *   3. On success, handleProcessingDone clears the session (PDFs no longer needed).
 *   4. On tab close/refresh, pagehide + beforeunload fire sendBeacon to clear the session.
 *   5. Backend hourly sweeper is the final backstop for anything missed.
 */
export default function App() {
    const [dark, setDark] = useState(true);
    const [nav, setNav] = useState("upload");
    const [screeningConfig, setScreeningConfig] = useState(null);
    const [results, setResults] = useState([]);
    const [activeModel, setActiveModel] = useState("mpnet");
    const [profileOpen, setProfileOpen] = useState(false);
    // Profile persisted to localStorage so it survives page reloads
    const [profile, setProfile] = useState(() => {
        try {
            const saved = localStorage.getItem("screeningProfile");
            return saved ? JSON.parse(saved) : { name: "", email: "", role: "", org: "" };
        } catch {
            return { name: "", email: "", role: "", org: "" };
        }
    });
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [backendOnline, setBackendOnline] = useState(null); // null=checking, true=online, false=offline

    // Poll /api/health with exponential backoff on failure.
    // BASE is a module constant; setBackendOnline is a stable setter — empty deps intentional.
    useEffect(() => {
        let intervalMs = 5000; // normal polling interval
        let timerId = null;
        const MAX_INTERVAL = 30000; // max 30s backoff
        const check = async () => {
            try {
                const ctrl = new AbortController();
                const t = setTimeout(() => ctrl.abort(), 2500);
                const res = await fetch(`${BASE}/api/health`, { signal: ctrl.signal });
                clearTimeout(t);
                setBackendOnline(res.ok);
                intervalMs = 5000; // reset on success
            } catch {
                setBackendOnline(false);
                intervalMs = Math.min(intervalMs * 1.5, MAX_INTERVAL); // backoff on failure
            }
            timerId = setTimeout(check, intervalMs);
        };
        check(); // run immediately on mount
        return () => { if (timerId) clearTimeout(timerId); };
    }, []); // BASE is a module constant — intentional empty deps

    // Save profile to state and persist to localStorage
    const handleSaveProfile = (newProfile) => {
        setProfile(newProfile);
        try {
            localStorage.setItem("screeningProfile", JSON.stringify(newProfile));
        } catch {
            // localStorage may be unavailable in private mode — fail silently
        }
    };

    // Sync module-level theme ref before render so all components read the right theme
    C = dark ? DARK : LIGHT;

    const width = useWindowWidth();
    const isMobile = width < 768;
    const isTablet = width < 1080;

    const initials = (profile.name || "U").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase() || "U";

    const navItems = [
        { id: "dashboard", icon: LayoutDashboard, label: "Dashboard" },
        { id: "upload", icon: Upload, label: "Upload" },
        { id: "config", icon: Briefcase, label: "Job Config" },
        { id: "candidates", icon: Users, label: "Candidates" },
        { id: "analytics", icon: BarChart2, label: "Analytics" },
    ];

    const go = useCallback((id) => { setNav(id); setSidebarOpen(false); }, []);

    // Listen for custom "navTo" DOM events dispatched by child views
    useEffect(() => {
        const handler = e => go(e.detail);
        document.addEventListener("navTo", handler);
        return () => document.removeEventListener("navTo", handler);
    }, [go]);

    // Tracks the session_id of the currently active upload — never held in
    // state (avoids re-renders); updated by ProcessingView via onSessionReady.
    const activeSessionRef = useRef(null);

    // Fire-and-forget DELETE of a session's uploads + result file from disk.
    // Called in three situations: new screening starts, results arrive, tab closes.
    const clearSession = useCallback((sid) => {
        if (!sid) return;
        fetch(`${BASE}/api/clear`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sid }),
        }).catch(() => {}); // swallow — cleanup is best-effort
    }, []);

    // On tab close / refresh / navigation away, fire sendBeacon so the browser
    // can dispatch the request even after the JS context is being torn down.
    // pagehide is preferred over beforeunload because it fires on bfcache too.
    useEffect(() => {
        const handleUnload = () => {
            const sid = activeSessionRef.current;
            if (!sid) return;
            const blob = new Blob(
                [JSON.stringify({ session_id: sid })],
                { type: "application/json" }
            );
            navigator.sendBeacon(`${BASE}/api/clear`, blob);
        };
        window.addEventListener("pagehide", handleUnload);
        window.addEventListener("beforeunload", handleUnload);
        return () => {
            window.removeEventListener("pagehide", handleUnload);
            window.removeEventListener("beforeunload", handleUnload);
        };
    }, []); // activeSessionRef is a ref — always read current value, no dep needed

    // Called by ProcessingView the moment /api/upload-resumes returns.
    // Lets App track this session for cleanup before screening even finishes.
    const handleSessionReady = useCallback((sid) => {
        activeSessionRef.current = sid;
    }, []);

    const handleStartScreening = useCallback((cfg) => {
        // Clear any leftover session from the previous run before starting a new one.
        if (activeSessionRef.current) {
            clearSession(activeSessionRef.current);
            activeSessionRef.current = null;
        }
        setScreeningConfig(cfg);
        setNav("processing");
        setSidebarOpen(false);
    }, [clearSession]);

    const handleProcessingDone = useCallback((apiResults) => {
        // Uploads are no longer needed once results are in memory — clear them now.
        // Result JSON stays in results/ on the backend; only the PDFs are removed.
        if (activeSessionRef.current) {
            clearSession(activeSessionRef.current);
            activeSessionRef.current = null;
        }
        const r = apiResults || [];
        setResults(r);
        // Only redirect to upload if nothing was processed at all.
        // If results exist but all are rejected, still go to dashboard.
        setNav(r.length > 0 ? "dashboard" : "upload");
    }, [clearSession]);

    // Switch active model, notify backend, and navigate to Upload for fresh screening
    const handleModelChange = async (key) => {
        if (key === activeModel) return;
        setActiveModel(key);
        go("upload");
        try {
            await fetch(`${BASE}/api/model`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: key }),
            });
        } catch {
            // Fail silently — model key is already updated in React state
        }
    };

    // Render the active page; switch prevents ProcessingView from remounting on unrelated state changes
    const renderView = () => {
        switch (nav) {
            case "dashboard":
                return <DashboardView results={results} onNav={go} isMobile={isMobile} activeModel={activeModel} onModelChange={handleModelChange} />;
            case "upload":
                return <UploadView onStartScreening={handleStartScreening} activeModel={activeModel} onModelChange={handleModelChange} isMobile={isMobile} backendOnline={backendOnline} />;
            case "processing":
                return <ProcessingView config={screeningConfig} onDone={handleProcessingDone} onSessionReady={handleSessionReady} />;
            case "config":
                return <JobConfigView isMobile={isMobile} />;
            case "candidates":
                return <CandidatesView results={results} onNav={go} isMobile={isMobile} />;
            case "analytics":
                return <AnalyticsView results={results} isMobile={isMobile} onNav={go} activeModel={activeModel} />;
            default:
                return <UploadView onStartScreening={handleStartScreening} activeModel={activeModel} onModelChange={handleModelChange} isMobile={isMobile} backendOnline={backendOnline} />;
        }
    };

    return (
        <>
            {/* ── Global CSS ─────────────────────────────────────────────── */}
            <style>{`
        *,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
        html { font-size:16px; }
        body { background:${C.bg}; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif; color:${C.text}; -webkit-font-smoothing:antialiased; font-weight:500; }
        ::-webkit-scrollbar { width:4px; height:4px; }
        ::-webkit-scrollbar-track { background:transparent; }
        ::-webkit-scrollbar-thumb { background:${C.scrollThumb}; border-radius:2px; }

        /* Animations */
        @keyframes fadeIn      { from{opacity:0}                to{opacity:1} }
        @keyframes fadeSlideUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }
        @keyframes slideInRight{ from{transform:translateX(100%)} to{transform:none} }
        @keyframes slideInLeft { from{transform:translateX(-100%)} to{transform:none} }
        @keyframes spin        { to{transform:rotate(360deg)} }
        @keyframes pulse       { 0%,100%{opacity:1} 50%{opacity:.35} }
        @keyframes runPulse    { 0%,100%{opacity:1} 50%{opacity:.6} }
        @keyframes runShimmer  { from{transform:translateX(-100%)} to{transform:translateX(200%)} }

        /* Profile modal — spring pop in, GPU accelerated */
        @keyframes modalIn {
          from { opacity:0; transform:translate3d(-50%,-48%,0) scale(.97) }
          to   { opacity:1; transform:translate3d(-50%,-50%,0) scale(1) }
        }

        /* Range sliders */
        input[type=range] { -webkit-appearance:none; appearance:none; height:4px; background:${C.inputBg}; border-radius:2px; width:100%; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; width:14px; height:14px; border-radius:50%; background:currentColor; cursor:pointer; }
        input[type=text]:focus, input[type=email]:focus, textarea:focus {
          outline: none;
          border-color: ${C.blue} !important;
          box-shadow: 0 0 0 2px ${C.blue}28 !important;
          transition: border-color .15s, box-shadow .15s;
        }
        textarea { color-scheme:${dark ? "dark" : "light"}; }
        button,input,textarea,select { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif; }
      `}</style>

            {/* ── Profile modal ──────────────────────────────────────────── */}
            {profileOpen && <ProfileModal profile={profile} onSave={handleSaveProfile} onClose={() => setProfileOpen(false)} />}

            {/* ── Mobile sidebar drawer ──────────────────────────────────── */}
            {isMobile && sidebarOpen && (
                <>
                    <div onClick={() => setSidebarOpen(false)} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,.55)", backdropFilter: "blur(4px)", zIndex: 300 }} />
                    <div style={{ position: "fixed", left: 0, top: 0, bottom: 0, width: 210, background: C.sidebarBg, borderRight: `1px solid ${C.border}`, zIndex: 310, padding: "20px 11px", display: "flex", flexDirection: "column", animation: "slideInLeft .24s ease" }}>
                        <SidebarContent navItems={navItems} nav={nav} go={go} backendOnline={backendOnline} results={results} />
                    </div>
                </>
            )}

            {/* ── Layout ─────────────────────────────────────────────────── */}
            <div style={{ display: "flex", minHeight: "100vh", background: C.bg, color: C.text }}>

                {/* Desktop sidebar */}
                {!isMobile && (
                    <aside style={{ width: isTablet ? 200 : 218, flexShrink: 0, borderRight: `1px solid ${C.border}`, display: "flex", flexDirection: "column", padding: "20px 11px", position: "sticky", top: 0, height: "100vh", background: C.sidebarBg }}>
                        <SidebarContent navItems={navItems} nav={nav} go={go} backendOnline={backendOnline} results={results} />
                    </aside>
                )}

                {/* Main content area */}
                <main style={{ flex: 1, display: "flex", flexDirection: "column", overflowY: "auto", overflowX: "hidden", maxHeight: "100vh", minWidth: 0 }}>

                    {/* Topbar */}
                    <div style={{ padding: isMobile ? "11px 14px" : "14px 24px", borderBottom: `1px solid ${C.border}`, display: "flex", justifyContent: "space-between", alignItems: "center", position: "sticky", top: 0, background: C.topbarBg, zIndex: 50, backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", gap: 8, minHeight: 64 }}>

                        {/* Left: hamburger (mobile) + page title */}
                        <div style={{ display: "flex", alignItems: "center", gap: 9, minWidth: 0 }}>
                            {isMobile && (
                                <button onClick={() => setSidebarOpen(true)} style={{ width: 30, height: 30, borderRadius: 7, border: `1px solid ${C.border}`, background: C.inputBg, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: C.sub, flexShrink: 0 }}>
                                    <Menu size={14} />
                                </button>
                            )}
                            <div style={{ minWidth: 0 }}>
                                <div style={{ fontSize: 10, color: C.muted, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".1em", whiteSpace: "nowrap" }}>
                                    {navItems.find(n => n.id === nav)?.label || "Processing"}
                                </div>
                                {!isMobile && (
                                    <div style={{ fontSize: 11, color: C.sub, marginTop: 1 }}>
                                        {new Date().toLocaleDateString("en-IN", { weekday: "long", month: "long", day: "numeric" })}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Right: model selector, theme toggle, profile */}
                        <div style={{ display: "flex", alignItems: "center", gap: 7, flexShrink: 0 }}>
                            <ModelDropdown activeModel={activeModel} onChange={handleModelChange} />

                            <button onClick={() => setDark(d => !d)} title="Toggle theme"
                                style={{ width: 30, height: 30, borderRadius: 7, border: `1px solid ${C.border}`, background: C.inputBg, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: C.sub, transition: "color .15s" }}
                                onMouseEnter={e => e.currentTarget.style.color = C.text}
                                onMouseLeave={e => e.currentTarget.style.color = C.sub}>
                                {dark ? <Sun size={13} /> : <Moon size={13} />}
                            </button>

                            <button onClick={() => setProfileOpen(true)} title={profile.name || "Edit Profile"}
                                style={{ width: 30, height: 30, borderRadius: 7, position: "relative", background: `linear-gradient(135deg,${C.blue},#a78bfa)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 800, color: "#fff", border: "none", cursor: "pointer", boxShadow: `0 2px 8px ${C.blue}44` }}>
                                {initials}
                                <div style={{ position: "absolute", bottom: -1, right: -1, width: 9, height: 9, borderRadius: "50%", background: C.blue, border: `2px solid ${C.topbarBg}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
                                    <Edit3 size={4} color="#fff" />
                                </div>
                            </button>
                        </div>
                    </div>

                    {/* Page body */}
                    <div style={{ flex: 1, padding: isMobile ? "12px 10px" : "22px 26px", animation: "fadeIn .18s ease" }} key={nav}>
                        {renderView()}
                    </div>
                </main>
            </div>
        </>
    );
}