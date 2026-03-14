import { useState, useRef, useCallback, useEffect } from "react";
import {
    LayoutDashboard, Briefcase, Users, BarChart2, Upload, X, Check, Plus,
    FileText, TrendingUp, Award, Mail, ChevronRight,
    Sparkles, Clock, Star, AlertCircle, CheckCircle, Zap, ArrowRight,
    RefreshCw, Sun, Moon, Edit3, Save, Activity, Target, Download,
    Menu, ChevronDown, Info, Phone,
} from "lucide-react";
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
    RadarChart, PolarGrid, PolarAngleAxis, Radar, CartesianGrid,
} from "recharts";

// ─── THEMES ───────────────────────────────────────────────────────────────────
const DARK = {
    bg: "#07070f", surface: "rgba(255,255,255,0.03)", border: "rgba(255,255,255,0.07)",
    text: "#e2e2f0", sub: "#6b6b8a", muted: "#484864",
    blue: "#818cf8", green: "#22c55e", amber: "#f59e0b", pink: "#f472b6", teal: "#34d399",
    sidebarBg: "rgba(255,255,255,0.015)", topbarBg: "#07070f",
    inputBg: "rgba(255,255,255,0.045)", cardText: "#c4c4d8",
    drawerBg: "#0d0d1b", scrollThumb: "rgba(255,255,255,0.08)",
    chartTooltip: "#0d0d1b", tableFocus: "rgba(129,140,248,0.05)",
    // Professional, non-neon chart fill colours
    chartSkill: "#7b93cc", chartSemantic: "#3d9e87", chartFinal: "#c98d3e",
};
const LIGHT = {
    bg: "#f1f3fa", surface: "#ffffff", border: "rgba(0,0,0,0.08)",
    text: "#111827", sub: "#6b7280", muted: "#9ca3af",
    blue: "#6366f1", green: "#16a34a", amber: "#d97706", pink: "#db2777", teal: "#0d9488",
    sidebarBg: "#e8eaf4", topbarBg: "#ffffff",
    inputBg: "rgba(0,0,0,0.04)", cardText: "#374151",
    drawerBg: "#f8f9fc", scrollThumb: "rgba(0,0,0,0.1)",
    chartTooltip: "#ffffff", tableFocus: "rgba(99,102,241,0.05)",
    // Professional, non-neon chart fill colours
    chartSkill: "#4a6fa5", chartSemantic: "#2a7c6b", chartFinal: "#b07030",
};

// Helper: derive a short display name that avoids truncating on a 1-2 char prefix
const shortDisplayName = (fullName, maxLen = 13) => {
    const parts = (fullName || "?").split(" ").filter(Boolean);
    let nm = parts[0] || "?";
    let pi = 1;
    while (nm.length <= 2 && pi < parts.length) { nm = parts.slice(0, pi + 1).join(" "); pi++; }
    return nm.slice(0, maxLen);
};

// Module-level theme ref — updated at top of each App render
let C = DARK;

// Helper: base card style with optional overrides
const card = (extra = {}) => ({
    background: C.surface, border: `1px solid ${C.border}`, borderRadius: 16, padding: 24, ...extra,
});

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const BASE = "http://localhost:5001";   // Flask backend URL

const PIPELINE_STEPS = (skillW = 55, modelName = "MPNet") => [
    { label: "resume_parser.py", desc: "Extracting text from PDFs via pdfplumber" },
    { label: "information_extractor.py", desc: "NER + regex — names, email, phone, skills, experience" },
    { label: "semantic_matcher.py", desc: `Generating sentence embeddings using ${modelName}` },
    { label: "semantic_matcher.py", desc: "Computing cosine similarity vs. job description" },
    { label: "scoring_engine.py", desc: "Applying eligibility filter (min experience check)" },
    { label: "scoring_engine.py", desc: `Weighted ranking — ${skillW}% skill / ${100 - skillW}% semantic` },
];

// Three supported embedding models — matches VALID_MODELS in app.py
const MODELS = {
    mpnet: {
        key: "mpnet",
        name: "MPNet",
        short: "multi-qa-mpnet-base-dot-v1",
        badge: "Best Accuracy",
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
        badge: "Top Ranking",
        color: "#38bdf8",
        desc: "Snowflake's enterprise embedding model — not SBERT, but a dedicated retrieval model optimised for high-precision document ranking and search at scale.",
        detail: "768-dim · MTEB top retrieval · enterprise-grade · ideal for high-precision screening.",
    },
};

// ─── HOOKS ────────────────────────────────────────────────────────────────────
function useWindowWidth() {
    const [w, setW] = useState(typeof window !== "undefined" ? window.innerWidth : 1200);
    useEffect(() => {
        const h = () => setW(window.innerWidth);
        window.addEventListener("resize", h);
        return () => window.removeEventListener("resize", h);
    }, []);
    return w;
}

// ─── SHARED PRIMITIVES ────────────────────────────────────────────────────────
const CircularRing = ({ value, size = 52, sw = 4.5, color }) => {
    const clr = color || C.blue;
    const r = (size - sw) / 2;
    const circ = r * 2 * Math.PI;
    const offset = circ - (value / 100) * circ;
    return (
        <div style={{ position: "relative", width: size, height: size, flexShrink: 0 }}>
            <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
                <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(128,128,128,0.13)" strokeWidth={sw} />
                <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={clr} strokeWidth={sw}
                    strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
                    style={{ transition: "stroke-dashoffset 1s cubic-bezier(.4,0,.2,1)" }} />
            </svg>
            <span style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: size * 0.22, fontWeight: 700, color: C.text }}>
                {value}
            </span>
        </div>
    );
};

const GaugeArc = ({ value }) => {
    const r = 64, sw = 12, cx = 85, cy = 82;
    const circ = Math.PI * r;
    const off = circ - (value / 100) * circ;
    const col = value >= 80 ? C.green : value >= 55 ? C.blue : C.amber;
    return (
        <svg width={170} height={90} viewBox="0 0 170 90">
            <path d={`M${cx - r} ${cy} A${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke="rgba(128,128,128,0.1)" strokeWidth={sw} strokeLinecap="round" />
            <path d={`M${cx - r} ${cy} A${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke={col} strokeWidth={sw} strokeLinecap="round"
                strokeDasharray={circ} strokeDashoffset={off}
                style={{ transition: "stroke-dashoffset 1.2s ease", filter: `drop-shadow(0 0 7px ${col}88)` }} />
            {/* Score centred vertically inside the arc */}
            <text x={cx} y={cy - 16} textAnchor="middle" fill={C.text} fontSize="26" fontWeight="900">{value}</text>
            <text x={cx} y={cy - 2} textAnchor="middle" fill={C.sub} fontSize="10">/ 100</text>
        </svg>
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
        <span style={{ padding: "3px 10px", borderRadius: 20, fontSize: 11, fontWeight: 600, background: s.bg, color: s.col, border: `1px solid ${s.col}40`, display: "inline-flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 5, height: 5, borderRadius: "50%", background: s.col }} />{status}
        </span>
    );
};

const SkillChip = ({ label, matched }) => (
    <span style={{
        padding: "3px 10px", borderRadius: 6, fontSize: 11, fontWeight: 500,
        background: matched ? `${C.blue}22` : C.inputBg,
        color: matched ? C.blue : C.sub,
        border: `1px solid ${matched ? `${C.blue}44` : C.border}`,
        display: "inline-flex", alignItems: "center", gap: 4,
    }}>
        {matched ? <Check size={9} /> : <X size={9} />} {label}
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

// ─── MODEL DROPDOWN ───────────────────────────────────────────────────────────
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
                padding: "6px 11px", borderRadius: 10,
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
                    position: "absolute", top: "calc(100% + 8px)", right: 0, width: 310,
                    borderRadius: 14, overflow: "hidden",
                    background: C.drawerBg, border: `1px solid ${C.border}`,
                    boxShadow: "0 16px 48px rgba(0,0,0,.35)", zIndex: 200,
                    animation: "fadeSlideUp .15s ease",
                }}>
                    <div style={{ padding: "10px 14px 7px", fontSize: 10, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".08em" }}>
                        Select Embedding Model
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

// ─── PROFILE MODAL ────────────────────────────────────────────────────────────
// Animation performance note:
// We use `will-change: transform` and `transform: translate3d(...)` to force
// GPU compositing, which eliminates the lag on first open.
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

// ─── CANDIDATE DRAWER ─────────────────────────────────────────────────────────
const Drawer = ({ candidate: c, onClose, isMobile }) => {
    if (!c) return null;
    const matched = c.matched_skills || [];
    const allSkills = c.skills || [];
    const missing = allSkills.filter(s => !matched.includes(s));
    const radarData = matched.slice(0, 6).map((s, i) => ({
        s,
        v: Math.round(55 + ((i * 37 + 13) % 40)),
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
                    {[
                        { icon: Mail, label: "Email", value: c.email || "Not Found", href: c.email ? `mailto:${c.email}` : null },
                        { icon: Clock, label: "Experience", value: !c.experience ? "Fresher" : `${c.experience} yrs`, href: null },
                        { icon: Phone, label: "Phone", value: c.phone || c.phone_number || c.contact || c.mobile || c.contact_number || "Not found", href: (c.phone || c.phone_number || c.contact || c.mobile || c.contact_number) ? `tel:${(c.phone || c.phone_number || c.contact || c.mobile || c.contact_number || "").replace(/\s/g, "")}` : null },
                    ].map(({ icon: Icon, label, value, href }) => (
                        <div key={label} style={{ display: "flex", gap: 10, padding: "9px 0", borderBottom: `1px solid ${C.border}`, alignItems: "flex-start" }}>
                            <Icon size={13} color={C.sub} style={{ marginTop: 2, flexShrink: 0 }} />
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
                    ))}
                    <div style={{ padding: "9px 0" }}>
                        <div style={{ fontSize: 10, color: C.sub, marginBottom: 5 }}>Status</div>
                        <Badge status={c.eligible ? "Shortlisted" : "Rejected"} />
                        {!c.eligible && c.rejection_reason && <div style={{ fontSize: 11, color: "#ef4444", marginTop: 5 }}>{c.rejection_reason}</div>}
                    </div>
                </div>

                {/* Radar chart */}
                {radarData.length >= 3 && (
                    <div style={card()}>
                        <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 10 }}>Skill Radar</div>
                        <ResponsiveContainer width="100%" height={220}>
                            <RadarChart data={radarData} outerRadius="68%">
                                <PolarGrid stroke={C.border} />
                                <PolarAngleAxis dataKey="s" tick={{ fill: C.sub, fontSize: 10 }} />
                                <Radar dataKey="v" stroke={C.chartSkill} fill={C.chartSkill} fillOpacity={0.18} strokeWidth={2} dot={{ fill: C.chartSkill, r: 3 }} />
                            </RadarChart>
                        </ResponsiveContainer>
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

// ─── UPLOAD VIEW ──────────────────────────────────────────────────────────────
const UploadView = ({ onStartScreening, activeModel, onModelChange, isMobile }) => {
    const [fileItems, setFileItems] = useState([]);
    const [isDragging, setIsDragging] = useState(false);
    const [toast, setToast] = useState(null);
    const [jd, setJd] = useState("");
    const [skills, setSkills] = useState([]);
    const [newSkill, setNewSkill] = useState("");
    const [minExp, setMinExp] = useState(0);
    const [skillWeight, setSkillWeight] = useState(55);
    const fileRef = useRef();
    const jdRef = useRef();

    const MAX_FILES = 30;
    const MAX_SIZE_MB = 10;

    // Pre-fill from saved config.json on mount
    useEffect(() => {
        fetch(`${BASE}/api/config`)
            .then(r => r.json())
            .then(data => {
                if (!data) return;
                if (data.job_description) setJd(data.job_description);
                if (data.required_skills?.length) setSkills(data.required_skills);
                if (data.scoring?.min_experience_years !== undefined)
                    setMinExp(data.scoring.min_experience_years);
                if (data.scoring?.skill_weight !== undefined)
                    setSkillWeight(Math.round(data.scoring.skill_weight * 100));
            })
            .catch(() => { });
    }, []);

    const showToast = (msg, type = "success") => {
        setToast({ msg, type });
        setTimeout(() => setToast(null), 3000);
    };

    const addFiles = useCallback((incoming) => {
        const arr = Array.from(incoming);
        const valid = arr.filter(f => f.type === "application/pdf");
        const bad = arr.filter(f => f.type !== "application/pdf");
        const tooBig = valid.filter(f => f.size > MAX_SIZE_MB * 1024 * 1024);
        const okSize = valid.filter(f => f.size <= MAX_SIZE_MB * 1024 * 1024);

        if (bad.length) showToast(`${bad.length} non-PDF file(s) skipped`, "error");
        if (tooBig.length) showToast(`${tooBig.length} file(s) exceed ${MAX_SIZE_MB}MB limit`, "error");
        if (!okSize.length) return;

        setFileItems(prev => {
            if (prev.length >= MAX_FILES) { showToast(`Max ${MAX_FILES} resumes per screening`, "error"); return prev; }
            const existing = new Set(prev.map(f => f.name));
            const fresh = okSize.filter(f => !existing.has(f.name)).map(f => ({
                id: `${f.name}-${Date.now()}`,
                name: f.name,
                raw: f,
                size: f.size > 1048576 ? (f.size / 1048576).toFixed(1) + " MB" : (f.size / 1024).toFixed(0) + " KB",
            }));
            if (fresh.length < okSize.length) showToast(`${okSize.length - fresh.length} duplicate(s) skipped`, "error");
            if (fresh.length) {
                showToast(`${fresh.length} PDF(s) added`, "success");
                // Auto-scroll to the JD section after a short delay
                setTimeout(() => jdRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 300);
            }
            return [...prev, ...fresh];
        });
    }, []);

    const onDrop = useCallback(e => {
        e.preventDefault(); setIsDragging(false); addFiles(e.dataTransfer.files);
    }, [addFiles]);

    const canStart = fileItems.length > 0 && jd.trim().length > 10;
    const activeM = MODELS[activeModel] || MODELS.mpnet;

    // Add skills — supports comma-separated and Enter
    const addSkill = (raw = newSkill) => {
        const parts = raw.split(",").map(s => s.trim().toLowerCase()).filter(s => s.length > 0);
        setSkills(prev => { const ex = new Set(prev); return [...prev, ...parts.filter(s => !ex.has(s))]; });
        setNewSkill("");
    };
    const handleSkillKeyDown = (e) => { if (e.key === "Enter") { e.preventDefault(); addSkill(); } };

    const handleStart = () => {
        if (!canStart) return;
        onStartScreening({ jd, skills, minExp, skillWeight, fileCount: fileItems.length, rawFiles: fileItems.map(f => f.raw), model: activeModel });
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
            <div>
                <h2 style={{ fontSize: 21, fontWeight: 800, color: C.text, margin: 0 }}>Upload & Configure</h2>
                <p style={{ color: C.sub, marginTop: 5, fontSize: 13 }}>Add PDF resumes and describe the role. The ML pipeline handles the rest.</p>
            </div>

            {/* Drop zone */}
            <div onClick={() => fileRef.current.click()}
                onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
                style={{ border: `2px dashed ${isDragging ? C.blue : C.border}`, borderRadius: 18, padding: "36px 28px", cursor: "pointer", textAlign: "center", background: isDragging ? `${C.blue}08` : C.inputBg, display: "flex", flexDirection: "column", alignItems: "center", gap: 10, transition: "all .2s" }}>
                <div style={{ width: 54, height: 54, borderRadius: 14, background: isDragging ? `${C.blue}20` : C.surface, border: `1px solid ${isDragging ? C.blue : C.border}`, display: "flex", alignItems: "center", justifyContent: "center", transition: "all .2s" }}>
                    <Upload size={22} color={isDragging ? C.blue : C.sub} />
                </div>
                <div>
                    <div style={{ fontSize: 15, fontWeight: 600, color: isDragging ? C.blue : C.cardText }}>
                        {isDragging ? "Drop to upload" : "Drag & drop PDF resumes here"}
                    </div>
                    <div style={{ fontSize: 12, color: C.sub, marginTop: 3 }}>
                        or click to browse · multiple files supported
                    </div>
                </div>
                {/* Limits notice */}
                <div style={{ display: "flex", gap: 12, marginTop: 2 }}>
                    {[`Max ${MAX_FILES} files`, `Max ${MAX_SIZE_MB} MB each`, "PDF format only"].map(t => (
                        <span key={t} style={{ fontSize: 10, color: C.muted, background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6, padding: "2px 8px" }}>{t}</span>
                    ))}
                </div>
                <input ref={fileRef} type="file" accept=".pdf" multiple hidden onChange={e => addFiles(e.target.files)} />
            </div>

            {/* Uploaded file list */}
            {fileItems.length > 0 && (
                <div style={card({ padding: 0, overflow: "hidden" })}>
                    <div style={{ padding: "10px 15px", borderBottom: `1px solid ${C.border}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".06em" }}>
                            {fileItems.length} file{fileItems.length !== 1 ? "s" : ""} queued
                        </span>
                        <button onClick={() => setFileItems([])} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 11, color: C.sub, fontFamily: "inherit" }}>Clear all</button>
                    </div>
                    {fileItems.map(f => (
                        <div key={f.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 15px", borderBottom: `1px solid ${C.border}22` }}>
                            <div style={{ width: 28, height: 28, borderRadius: 7, background: "rgba(239,68,68,.1)", border: "1px solid rgba(239,68,68,.18)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                                <FileText size={12} color="#ef4444" />
                            </div>
                            <div style={{ flex: 1, minWidth: 0 }}>
                                <div style={{ fontSize: 13, color: C.cardText, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.name}</div>
                                <div style={{ fontSize: 10, color: C.sub }}>{f.size}</div>
                            </div>
                            <CheckCircle size={12} color={C.green} />
                            <button onClick={() => setFileItems(p => p.filter(x => x.id !== f.id))} style={{ background: "none", border: "none", cursor: "pointer", color: C.sub }}>
                                <X size={11} />
                            </button>
                        </div>
                    ))}
                </div>
            )}

            {/* Job requirements — two columns on desktop, same height */}
            <div ref={jdRef} style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr", gap: 18, alignItems: "stretch" }}>
                {/* Left column: JD */}
                <div style={{ ...card(), display: "flex", flexDirection: "column" }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 14, display: "flex", alignItems: "center", gap: 7 }}>
                        <Briefcase size={12} /> Job Description
                    </div>
                    <label style={{ fontSize: 12, color: C.sub, display: "block", marginBottom: 6 }}>Describe the Role</label>
                    <textarea value={jd} onChange={e => { setJd(e.target.value); e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; }}
                        placeholder="Describe the role — what the candidate will be doing and what you expect from them."
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
                            placeholder="python, machine learning, sql  — or Enter"
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

            {/*
              ── MODEL INFO PANEL + SWITCHER ───────────────────────────────────
            */}
            <div style={{
                borderRadius: 16, overflow: "hidden",
                border: `1px solid ${activeM.color}30`,
                background: `${activeM.color}06`,
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
                    <span style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".07em", flexShrink: 0 }}>Switch Model:</span>
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
                        <style>{`@keyframes runPulse{0%,100%{opacity:1}50%{opacity:.6}} @keyframes runShimmer{from{transform:translateX(-100%)}to{transform:translateX(200%)}}`}</style>
                        {/* shimmer */}
                        <span style={{ position: "absolute", inset: 0, background: "linear-gradient(90deg,transparent,rgba(255,255,255,.18),transparent)", animation: "runShimmer 2.2s ease infinite", pointerEvents: "none" }} />
                        <Zap size={16} style={{ animation: "runPulse 1.6s ease infinite" }} />
                        Run Screening with {activeM.name}
                        <ArrowRight size={15} />
                    </button>
                ) : (
                    <div style={{ padding: "13px 20px", display: "flex", alignItems: "center", justifyContent: "center", gap: 9, fontSize: 12, color: C.sub }}>
                        <Info size={13} color={C.sub} style={{ flexShrink: 0 }} />
                        {fileItems.length === 0
                            ? "Upload at least one resume PDF to continue."
                            : "Add a job description (min 10 characters) to continue."}
                    </div>
                )}
            </div>

            {toast && <Toast msg={toast.msg} type={toast.type} onClose={() => setToast(null)} />}
        </div>
    );
};

// ─── PROCESSING VIEW ──────────────────────────────────────────────────────────
const ProcessingView = ({ config, onDone }) => {
    const [progress, setProgress] = useState(0);
    const [step, setStep] = useState(0);
    const [status, setStatus] = useState("Connecting to backend…");
    const [error, setError] = useState(null);
    const doneRef = useRef(false);
    const ivRef = useRef(null);

    useEffect(() => {
        if (doneRef.current) return;

        // Animated progress bar — slows near 88% while waiting for the API response
        let current = 0;
        ivRef.current = setInterval(() => {
            current += current < 85 ? 2.0 : 0.1;
            const clamped = Math.min(current, 99);
            setProgress(clamped);
            setStep(Math.min(Math.floor(clamped / 17), PIPELINE_STEPS(config?.skillWeight || 55, MODELS[config?.model || "mpnet"]?.name || "MPNet").length - 1));
        }, 65);

        const run = async () => {
            try {
                // 1. Upload PDFs
                setStatus("Uploading PDFs to backend…");
                const fd = new FormData();
                (config.rawFiles || []).forEach(f => fd.append("files", f));
                const upRes = await fetch(`${BASE}/api/upload-resumes`, { method: "POST", body: fd });
                if (!upRes.ok) {
                    const e = await upRes.json();
                    throw new Error(e.error || "Upload failed");
                }

                // 2. Run ML pipeline
                setStatus("Running ML pipeline — this may take a moment…");
                const screenRes = await fetch(`${BASE}/api/screen`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        job_description: config.jd,
                        required_skills: config.skills || [],
                        model: config.model || "mpnet",
                        config: {
                            min_experience_years: config.minExp || 0,
                            skill_weight: (config.skillWeight || 55) / 100,
                            semantic_weight: 1 - (config.skillWeight || 55) / 100,
                        },
                    }),
                });
                const data = await screenRes.json();
                if (!screenRes.ok) throw new Error(data.error || "Screening failed");

                // Done
                clearInterval(ivRef.current);
                doneRef.current = true;
                setProgress(100);
                setStep(PIPELINE_STEPS(config?.skillWeight || 55, MODELS[config?.model || "mpnet"]?.name || "MPNet").length - 1);
                setStatus("Complete!");
                setTimeout(() => onDone(data.results || []), 700);

            } catch (err) {
                clearInterval(ivRef.current);
                setError(err.message);
            }
        };

        run();
        return () => clearInterval(ivRef.current);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const pct = Math.round(progress);
    const modelInfo = MODELS[config?.model || "mpnet"];

    // Error screen
    if (error) {
        return (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "60vh", gap: 14, textAlign: "center", padding: "24px 16px" }}>
                <div style={{ width: 54, height: 54, borderRadius: 15, background: "rgba(239,68,68,.12)", border: "1px solid rgba(239,68,68,.25)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <AlertCircle size={22} color="#ef4444" />
                </div>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.text }}>Pipeline Error</div>
                <div style={{ fontSize: 13, color: "#ef4444", maxWidth: 380, lineHeight: 1.6, background: "rgba(239,68,68,.08)", padding: "10px 16px", borderRadius: 10, border: "1px solid rgba(239,68,68,.2)", wordBreak: "break-word" }}>{error}</div>
                <div style={{ fontSize: 12, color: C.sub }}>Ensure Flask is running: <code style={{ color: C.blue }}>python app.py</code></div>
                <button onClick={() => onDone([])} style={{ padding: "9px 20px", borderRadius: 9, border: `1px solid ${C.border}`, background: C.inputBg, color: C.sub, cursor: "pointer", fontSize: 13, fontFamily: "inherit", fontWeight: 600 }}>
                    ← Back
                </button>
            </div>
        );
    }

    return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "calc(100vh - 130px)", padding: "24px 14px" }}>
            <div style={{ width: "100%", maxWidth: 520 }}>
                <div style={{ textAlign: "center", marginBottom: 28 }}>
                    <div style={{ fontSize: 20, fontWeight: 800, color: C.text, marginBottom: 5 }}>Running ML Pipeline</div>
                    <div style={{ fontSize: 13, color: C.sub }}>
                        Analysing <strong style={{ color: C.blue }}>{config?.fileCount || 0}</strong> resume(s) · {status}
                    </div>
                </div>

                {/* Ring + bar */}
                <div style={card({ display: "flex", flexDirection: "column", alignItems: "center", gap: 18, marginBottom: 12 })}>
                    <div style={{ position: "relative", width: 112, height: 112 }}>
                        <svg width={112} height={112} style={{ transform: "rotate(-90deg)" }}>
                            <circle cx={56} cy={56} r={48} fill="none" stroke={`${modelInfo.color}15`} strokeWidth={8} />
                            <circle cx={56} cy={56} r={48} fill="none" stroke={modelInfo.color} strokeWidth={8} strokeLinecap="round"
                                strokeDasharray={2 * Math.PI * 48}
                                strokeDashoffset={2 * Math.PI * 48 * (1 - progress / 100)}
                                style={{ transition: "stroke-dashoffset .12s linear", filter: `drop-shadow(0 0 10px ${modelInfo.color}80)` }}
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

                {/* Pipeline steps */}
                <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                    {PIPELINE_STEPS(config?.skillWeight || 55, MODELS[config?.model || "mpnet"]?.name || "MPNet").map((s, i) => {
                        const done = i < step;
                        const active = i === step;
                        return (
                            <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 13px", borderRadius: 10, background: done ? `${C.green}08` : active ? `${modelInfo.color}0f` : C.inputBg, border: `1px solid ${done ? `${C.green}20` : active ? `${modelInfo.color}26` : C.border}`, transition: "all .3s" }}>
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

// ─── DASHBOARD VIEW ───────────────────────────────────────────────────────────
const DashboardView = ({ results, onNav, isMobile, activeModel, onModelChange }) => {
    if (!results || results.length === 0) {
        return (
            <EmptyState icon={LayoutDashboard} title="No results yet"
                sub="Upload resumes and run the ML screening pipeline to see ranked candidates here."
                action={
                    <button onClick={() => onNav("upload")} style={{ padding: "9px 20px", borderRadius: 10, border: "none", background: `linear-gradient(135deg,${C.blue},#6366f1)`, color: "#fff", cursor: "pointer", fontSize: 13, fontWeight: 700, fontFamily: "inherit", display: "flex", alignItems: "center", gap: 7 }}>
                        <Upload size={13} /> Upload Resumes
                    </button>
                }
            />
        );
    }

    const eligible = results.filter(c => c.eligible);
    const top = eligible[0] || results[0];
    const avgScore = eligible.length ? Math.round(eligible.reduce((a, c) => a + (c.finalScore || 0), 0) / eligible.length * 100) : 0;
    const scoreDist = eligible.slice(0, 8).map(c => ({
        name: shortDisplayName(c.name),
        Skill: Math.round((c.skillScore || 0) * 100),
        Semantic: Math.round((c.semanticScore || 0) * 100),
        Final: Math.round((c.finalScore || 0) * 100),
    }));
    const cols4 = isMobile ? "1fr 1fr" : "repeat(4,1fr)";
    const cols2 = isMobile ? "1fr" : "1fr 1.65fr";

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* ── Stats row ───────────────────────────────────────────────── */}
            <div style={{ display: "grid", gridTemplateColumns: cols4, gap: 12 }}>
                {[
                    { icon: FileText, label: "Screened", value: results.length, sub: "Resumes processed", color: C.blue },
                    { icon: Award, label: "Shortlisted", value: eligible.length, sub: "Eligible candidates", color: C.green },
                    { icon: TrendingUp, label: "Avg Score", value: `${avgScore}%`, sub: "Across eligible", color: C.amber },
                    { icon: Star, label: "Top Score", value: `${Math.round((top.finalScore || 0) * 100)}%`, sub: top.name || "—", color: C.pink },
                ].map(({ icon: Icon, label, value, sub, color }) => (
                    <div key={label} style={card({ display: "flex", gap: 11, alignItems: "flex-start", padding: 16 })}>
                        <div style={{ width: 36, height: 36, borderRadius: 9, background: `${color}18`, border: `1px solid ${color}25`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                            <Icon size={15} color={color} />
                        </div>
                        <div style={{ minWidth: 0 }}>
                            <div style={{ fontSize: 11, color: C.sub, marginBottom: 1 }}>{label}</div>
                            <div style={{ fontSize: 20, fontWeight: 800, color: C.text, lineHeight: 1.1 }}>{value}</div>
                            <div style={{ fontSize: 10, color: C.muted, marginTop: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{sub}</div>
                        </div>
                    </div>
                ))}
            </div>

            {/* ── Gauge + bar chart ──────────────────────────────────────── */}
            {eligible.length > 0 && (
                <div style={{ display: "grid", gridTemplateColumns: cols2, gap: 12 }}>
                    <div style={card({ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 18 })}>
                        <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 10 }}>Top Candidate</div>
                        <GaugeArc value={Math.round((top.finalScore || 0) * 100)} />
                        <div style={{ fontSize: 13, fontWeight: 700, color: C.text, marginTop: 4 }}>{top.name}</div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 5, justifyContent: "center", marginTop: 9 }}>
                            {(top.matched_skills || []).slice(0, 4).map(s => <SkillChip key={s} label={s} matched />)}
                        </div>
                    </div>
                    <div style={card({ padding: 18 })}>
                        {/* Header row */}
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                            <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em" }}>Score Distribution</div>
                            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                <div style={{ fontSize: 10, color: C.muted }}>Top {scoreDist.length} candidates</div>
                                <button onClick={() => onNav("analytics")} style={{ fontSize: 10, color: C.blue, background: `${C.blue}12`, border: `1px solid ${C.blue}25`, borderRadius: 8, padding: "2px 9px", cursor: "pointer", fontFamily: "inherit", fontWeight: 600 }}>
                                    More →
                                </button>
                            </div>
                        </div>
                        <ResponsiveContainer width="100%" height={210}>
                            <BarChart data={scoreDist} barSize={8} barGap={2}
                                margin={{ top: 4, right: 4, bottom: 8, left: 4 }}>
                                <XAxis dataKey="name"
                                    tick={{ fill: C.sub, fontSize: 9 }}
                                    axisLine={false} tickLine={false}
                                    interval={0} />
                                <YAxis hide domain={[0, 100]} />
                                <Tooltip
                                    cursor={{ fill: `${C.blue}08`, radius: 4 }}
                                    contentStyle={{ background: C.drawerBg, border: `1px solid ${C.border}`, borderRadius: 10, fontSize: 11, backdropFilter: "blur(10px)", boxShadow: "0 8px 32px rgba(0,0,0,.35)", color: C.text }}
                                    labelStyle={{ color: C.sub, fontWeight: 700 }} />
                                <Bar dataKey="Skill" fill={C.chartSkill} radius={[3, 3, 0, 0]} />
                                <Bar dataKey="Semantic" fill={C.chartSemantic} radius={[3, 3, 0, 0]} />
                                <Bar dataKey="Final" fill={C.chartFinal} radius={[3, 3, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                        <div style={{ display: "flex", gap: 12, justifyContent: "center", marginTop: 4 }}>
                            {[["Skill", C.chartSkill], ["Semantic", C.chartSemantic], ["Final", C.chartFinal]].map(([l, c]) => (
                                <div key={l} style={{ display: "flex", gap: 5, alignItems: "center", fontSize: 10, color: C.sub }}>
                                    <div style={{ width: 7, height: 7, borderRadius: 2, background: c }} />{l}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* ── Quick Actions — between scores and About panel ──────── */}
            <div style={card({ padding: 16 })}>
                <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 11 }}>Quick Actions</div>
                <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr 1fr" : "repeat(4,1fr)", gap: 9 }}>
                    {[
                        { label: "View Rankings", icon: Users, nav: "candidates", color: C.blue },
                        { label: "Analytics", icon: BarChart2, nav: "analytics", color: C.teal },
                        { label: "Job Config", icon: Briefcase, nav: "config", color: C.amber },
                        { label: "New Screening", icon: RefreshCw, nav: "upload", color: C.pink },
                    ].map(({ label, icon: Icon, nav, color }) => (
                        <button key={nav} onClick={() => onNav(nav)} style={{ padding: "10px 11px", borderRadius: 10, background: `${color}0e`, border: `1px solid ${color}20`, color, fontSize: 12, fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", gap: 6, transition: "all .15s", fontFamily: "inherit" }}>
                            <Icon size={12} /> {label}
                        </button>
                    ))}
                </div>
            </div>

            {/* ── About this System / Developer panel ───────────────────── */}
            <div style={card({ padding: 22, background: `linear-gradient(135deg,rgba(129,140,248,.06),rgba(52,211,153,.04))`, borderColor: `${C.blue}25` })}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 18 }}>
                    <div style={{ width: 38, height: 38, borderRadius: 10, background: `linear-gradient(135deg,${C.blue},#a78bfa)`, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: `0 4px 14px ${C.blue}44` }}>
                        <Sparkles size={16} color="#fff" />
                    </div>
                    <div>
                        <div style={{ fontSize: 15, fontWeight: 800, color: C.text }}>ML-Based Resume Screening System</div>
                        <div style={{ fontSize: 11, color: C.sub }}>Developed by <span style={{ color: C.blue, fontWeight: 700 }}>Madhan Kumar</span> · B.Sc Computer Science Final Year Project</div>
                    </div>
                </div>
                <p style={{ fontSize: 13, color: C.cardText, lineHeight: 1.7, marginBottom: 18 }}>
                    This system uses <strong style={{ color: C.blue }}>transformer-based sentence embedding models</strong> to semantically match resumes against job descriptions — going far beyond simple keyword matching. Each resume is converted into a high-dimensional vector using the selected model, and <strong style={{ color: C.teal }}>cosine similarity</strong> is computed against the job description embedding. Combined with a skill-coverage score, candidates are ranked using a <strong style={{ color: C.amber }}>configurable weighted formula</strong> — the exact skill vs. semantic split is set by you using the Skill Weight slider in the Upload tab.
                </p>
                <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "repeat(3,1fr)", gap: 11 }}>
                    {Object.values(MODELS).map(m => (
                        <div key={m.key} style={{ padding: "13px 15px", borderRadius: 12, background: `${m.color}08`, border: `1px solid ${m.color}${activeModel === m.key ? "50" : "20"}`, position: "relative" }}>
                            {activeModel === m.key && (
                                <div style={{ position: "absolute", top: 9, right: 9, padding: "1px 7px", borderRadius: 10, fontSize: 9, fontWeight: 700, background: `${m.color}25`, color: m.color }}>● Active</div>
                            )}
                            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                                <div style={{ width: 8, height: 8, borderRadius: "50%", background: m.color, boxShadow: `0 0 6px ${m.color}` }} />
                                <span style={{ fontSize: 13, fontWeight: 800, color: C.text }}>{m.name}</span>
                                <span style={{ padding: "1px 7px", borderRadius: 10, fontSize: 9, fontWeight: 700, background: `${m.color}18`, color: m.color }}>{m.badge}</span>
                            </div>
                            <div style={{ fontSize: 10, fontFamily: "monospace", color: C.sub, marginBottom: 5, wordBreak: "break-all" }}>{m.short}</div>
                            <div style={{ fontSize: 11, color: C.sub, lineHeight: 1.5 }}>{m.desc}</div>
                            <div style={{ fontSize: 10, color: C.muted, marginTop: 4, fontStyle: "italic" }}>{m.detail}</div>
                        </div>
                    ))}
                </div>
            </div>

        </div>
    );
};

// Rank medal colors
const RANK_COLORS = {
    1: { bg: "rgba(255,215,0,.18)", border: "rgba(255,215,0,.45)", text: "#FFD700" },   // Gold
    2: { bg: "rgba(192,192,192,.18)", border: "rgba(192,192,192,.45)", text: "#C0C0C0" },   // Silver
    3: { bg: "rgba(205,127,50,.18)", border: "rgba(205,127,50,.45)", text: "#CD7F32" },   // Bronze
};

// Export shortlisted candidates — rank, name, email, phone only
function exportToCSV(results) {
    const shortlisted = results.filter(c => c.eligible);
    if (!shortlisted.length) return;

    const headers = ["Rank", "Name", "Email", "Phone"];
    const rows = shortlisted.map(c => [
        c.rank,
        `"${(c.name || "").replace(/"/g, "'")}"`,
        `"${(c.email || "").replace(/"/g, "'")}"`,
        `"${(c.phone || c.phone_number || "").replace(/"/g, "'")}"`,
    ]);

    const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "shortlisted_candidates.csv";
    a.click();
    URL.revokeObjectURL(url);
}

const CandidatesView = ({ results, onNav, isMobile }) => {
    const [selected, setSelected] = useState(null);
    const [filter, setFilter] = useState("All");

    if (!results || results.length === 0) {
        return (
            <EmptyState icon={Users} title="No candidates yet"
                sub="Run the screening pipeline first to see ranked candidates here."
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
                    <p style={{ color: C.sub, marginTop: 3, fontSize: 12 }}>Click any row to view full profile and skill gap analysis.</p>
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
                                    style={{ borderBottom: `1px solid ${C.border}22`, cursor: "pointer", transition: "background .12s" }}
                                    onMouseEnter={e => e.currentTarget.style.background = C.tableFocus}
                                    onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
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
                                    <td style={{ padding: "12px 15px" }}><ChevronRight size={13} color={C.muted} /></td>
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

// ─── JOB CONFIG VIEW ──────────────────────────────────────────────────────────
const JobConfigView = () => {
    const [saved, setSaved] = useState(false);
    const [cfg, setCfg] = useState({ jd: "", skills: [], minExp: 0, skillW: 55, semanticW: 45 });
    const [newSkill, setNewSkill] = useState("");

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
            .catch(() => { });
    }, []);

    const addCfgSkill = (raw = newSkill) => {
        const parts = raw.split(",").map(s => s.trim().toLowerCase()).filter(s => s.length > 0);
        setCfg(prev => { const ex = new Set(prev.skills); return { ...prev, skills: [...prev.skills, ...parts.filter(s => !ex.has(s))] }; });
        setNewSkill("");
    };

    const save = async () => {
        try {
            await fetch(`${BASE}/api/config`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    job_description: cfg.jd,
                    required_skills: cfg.skills,
                    scoring: {
                        skill_weight: cfg.skillW / 100,
                        semantic_weight: cfg.semanticW / 100,
                        min_experience_years: cfg.minExp,
                        top_n: 20,
                    },
                }),
            });
        } catch (_) { }
        setSaved(true);
        setTimeout(() => setSaved(false), 2500);
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
                    <span style={{ fontSize: 12, fontWeight: 700, color: C.blue }}>What is Job Config for?</span>
                </div>
                <p style={{ fontSize: 12, color: C.cardText, lineHeight: 1.7, margin: 0 }}>
                    Think of this as your <strong style={{ color: C.text }}>saved job profile</strong>. When you screen multiple batches of resumes for the same role over several days, you don't need to retype the job description and skills each time — just save them here once and they auto-fill in the Upload tab. It also lets you standardise the scoring weights (skill vs. semantic) for your organisation.
                </p>
            </div>

            {/* Two-column layout same as Upload */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18, alignItems: "stretch" }}>
                {/* Left: JD */}
                <div style={{ ...card(), display: "flex", flexDirection: "column" }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 14, display: "flex", alignItems: "center", gap: 7 }}>
                        <Briefcase size={12} /> Job Description
                    </div>
                    <textarea value={cfg.jd} onChange={e => { setCfg(p => ({ ...p, jd: e.target.value })); e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; }}
                        placeholder="Describe the role — what the candidate will be doing and what you expect from them."
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
                            placeholder="python, machine learning, sql  — or Enter"
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

            {/* Centered save button */}
            <div style={{ display: "flex", justifyContent: "center" }}>
                <button onClick={save} style={{ padding: "12px 36px", borderRadius: 12, border: `1px solid ${saved ? `${C.green}40` : `${C.blue}40`}`, background: saved ? `${C.green}12` : `linear-gradient(135deg,${C.blue}18,#a78bfa18)`, color: saved ? C.green : C.blue, cursor: "pointer", fontSize: 14, fontWeight: 700, display: "flex", alignItems: "center", gap: 9, fontFamily: "inherit", transition: "all .3s", boxShadow: saved ? `0 0 16px ${C.green}22` : `0 0 16px ${C.blue}18` }}>
                    {saved ? <><CheckCircle size={15} /> Saved to config.json</> : <><Save size={15} /> Save Configuration</>}
                </button>
            </div>
        </div>
    );
};

// ─── SKILL COVERAGE ANALYSIS ───────────────────────────────────────────────────
const SkillCoverageAnalysis = ({ results, isMobile }) => {
    const skillCount = {};
    const skillTotal = results.length || 1;
    results.forEach(c => (c.matched_skills || []).forEach(s => { skillCount[s] = (skillCount[s] || 0) + 1; }));

    const missingCount = {};
    results.forEach(c => (c.missing_skills || []).forEach(s => { missingCount[s] = (missingCount[s] || 0) + 1; }));

    const allSkills = new Set([...Object.keys(skillCount), ...Object.keys(missingCount)]);
    const skillData = [...allSkills].map(skill => ({
        skill,
        matched: skillCount[skill] || 0,
        missing: missingCount[skill] || 0,
        coverage: Math.round(((skillCount[skill] || 0) / skillTotal) * 100),
    })).sort((a, b) => b.coverage - a.coverage);

    if (skillData.length === 0) return null;

    return (
        <div style={card({ padding: 18 })}>
            <div style={{ marginBottom: 14 }}>
                <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 4 }}>
                    Skill Coverage Analysis
                </div>
                <div style={{ fontSize: 12, color: C.cardText, lineHeight: 1.6 }}>
                    Shows how many candidates match each required skill. Longer bars mean the skill is common in the talent pool; short bars reveal hard-to-find skills.
                </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {skillData.slice(0, isMobile ? 10 : 15).map(({ skill, matched, coverage }) => {
                    const barCol = coverage >= 70 ? C.green : coverage >= 40 ? C.blue : coverage >= 20 ? C.amber : "#e74c5e";
                    return (
                        <div key={skill} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                            <div style={{ width: isMobile ? 90 : 130, fontSize: 11, fontWeight: 600, color: C.text, textAlign: "right", flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                {skill}
                            </div>
                            <div style={{ flex: 1, height: 18, background: C.inputBg, borderRadius: 6, overflow: "hidden", position: "relative" }}>
                                <div style={{ width: `${Math.max(coverage, 2)}%`, height: "100%", background: `${barCol}cc`, borderRadius: 6, transition: "width .4s ease" }} />
                            </div>
                            <div style={{ width: 60, fontSize: 11, fontWeight: 700, color: barCol, textAlign: "right", flexShrink: 0 }}>
                                {matched}/{results.length}
                                <span style={{ color: C.muted, fontWeight: 400, marginLeft: 3 }}>({coverage}%)</span>
                            </div>
                        </div>
                    );
                })}
            </div>

            {skillData.length > (isMobile ? 10 : 15) && (
                <div style={{ fontSize: 10, color: C.muted, marginTop: 8, textAlign: "center" }}>
                    Showing top {isMobile ? 10 : 15} of {skillData.length} skills
                </div>
            )}
        </div>
    );
};

// ─── ANALYTICS VIEW ───────────────────────────────────────────────────────────
const AnalyticsView = ({ results, isMobile }) => {
    if (!results || results.length === 0) {
        return <EmptyState icon={BarChart2} title="No analytics data" sub="Run the screening pipeline to generate analytics." />;
    }

    const eligible = results.filter(c => c.eligible);
    const rejected = results.filter(c => !c.eligible);
    const top = eligible[0];
    const avgFinal = eligible.length ? Math.round(eligible.reduce((a, c) => a + (c.finalScore || 0), 0) / eligible.length * 100) : 0;
    const avgSkill = eligible.length ? Math.round(eligible.reduce((a, c) => a + (c.skillScore || 0), 0) / eligible.length * 100) : 0;
    const avgSem = eligible.length ? Math.round(eligible.reduce((a, c) => a + (c.semanticScore || 0), 0) / eligible.length * 100) : 0;

    // Vertical score distribution — all candidates, sorted by final score
    const scoreDist = [...results]
        .sort((a, b) => (b.finalScore || 0) - (a.finalScore || 0))
        .map(c => ({
            name: shortDisplayName(c.name),
            Final: Math.round((c.finalScore || 0) * 100),
            Skill: Math.round((c.skillScore || 0) * 100),
            Semantic: Math.round((c.semanticScore || 0) * 100),
        }));

    // Skill frequency — how many candidates matched each skill (available for future extension)
    const skillCount = {};
    results.forEach(c => (c.matched_skills || []).forEach(s => { skillCount[s] = (skillCount[s] || 0) + 1; }));
    // eslint-disable-next-line no-unused-vars
    const _skillFreq = Object.entries(skillCount)
        .map(([skill, count]) => ({ skill, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10);

    // Score buckets — distribution of candidates by score range
    const buckets = { "0–20": 0, "21–40": 0, "41–60": 0, "61–80": 0, "81–100": 0 };
    results.forEach(c => {
        const s = Math.round((c.finalScore || 0) * 100);
        if (s <= 20) buckets["0–20"]++;
        else if (s <= 40) buckets["21–40"]++;
        else if (s <= 60) buckets["41–60"]++;
        else if (s <= 80) buckets["61–80"]++;
        else buckets["81–100"]++;
    });

    const topRadar = top ? (top.matched_skills || []).slice(0, 6).map((s, i) => ({
        s,
        v: Math.round(55 + ((i * 37 + 13) % 40)),
    })) : [];

    const TOOLTIP = { background: C.drawerBg, border: `1px solid ${C.border}`, borderRadius: 10, fontSize: 11, backdropFilter: "blur(10px)", boxShadow: "0 8px 32px rgba(0,0,0,.35)", color: C.text };
    const LABEL = { color: C.sub, fontWeight: 700 };

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div>
                <h2 style={{ fontSize: 20, fontWeight: 800, color: C.text, margin: 0 }}>Analytics</h2>
                <p style={{ color: C.sub, marginTop: 4, fontSize: 12 }}>Deep-dive into scores, skill coverage, and candidate performance.</p>
            </div>

            {/* ── KPI strip ─────────────────────────────────────────────── */}
            <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr 1fr" : "repeat(5,1fr)", gap: 10 }}>
                {[
                    { label: "Total Screened", value: results.length, color: C.blue },
                    { label: "Shortlisted", value: eligible.length, color: C.green },
                    { label: "Rejected", value: rejected.length, color: "#ef4444" },
                    { label: "Avg Final Score", value: `${avgFinal}%`, color: C.amber },
                    { label: "Pass Rate", value: `${results.length ? Math.round(eligible.length / results.length * 100) : 0}%`, color: C.teal },
                ].map(({ label, value, color }) => (
                    <div key={label} style={card({ padding: "13px 16px", textAlign: "center" })}>
                        <div style={{ fontSize: 22, fontWeight: 900, color, lineHeight: 1 }}>{value}</div>
                        <div style={{ fontSize: 10, color: C.sub, marginTop: 4 }}>{label}</div>
                    </div>
                ))}
            </div>

            {/* ── Vertical score distribution + candidate ranking table ── */}
            <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr", gap: 13 }}>

                {/* Vertical bar chart */}
                <div style={card({ padding: 18 })}>
                    <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 14 }}>Score Distribution — All Candidates</div>
                    <ResponsiveContainer width="100%" height={Math.max(220, scoreDist.length * 28)}>
                        <BarChart data={scoreDist} layout="vertical" barSize={7} barGap={2} margin={{ left: 0, right: 10 }}>
                            <CartesianGrid stroke={C.border} horizontal={false} />
                            <XAxis type="number" domain={[0, 100]} tick={{ fill: C.sub, fontSize: 10 }} axisLine={false} tickLine={false} />
                            <YAxis type="category" dataKey="name" tick={{ fill: C.sub, fontSize: 10 }} axisLine={false} tickLine={false} width={60} />
                            <Tooltip cursor={{ fill: `${C.blue}08` }} contentStyle={TOOLTIP} labelStyle={LABEL} />
                            <Bar dataKey="Skill" fill={C.chartSkill} radius={[0, 3, 3, 0]} />
                            <Bar dataKey="Semantic" fill={C.chartSemantic} radius={[0, 3, 3, 0]} />
                            <Bar dataKey="Final" fill={C.chartFinal} radius={[0, 3, 3, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                    <div style={{ display: "flex", gap: 12, justifyContent: "center", marginTop: 8 }}>
                        {[["Skill", C.chartSkill], ["Semantic", C.chartSemantic], ["Final", C.chartFinal]].map(([l, c]) => (
                            <div key={l} style={{ display: "flex", gap: 5, alignItems: "center", fontSize: 10, color: C.sub }}>
                                <div style={{ width: 7, height: 7, borderRadius: 2, background: c }} />{l}
                            </div>
                        ))}
                    </div>
                </div>

                {/* Candidate ranking table — all shortlisted */}
                <div style={card({ padding: 18, display: "flex", flexDirection: "column" })}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                        <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em" }}>Candidate Final Rankings</div>
                        <button
                            onClick={() => document.dispatchEvent(new CustomEvent("navTo", { detail: "candidates" }))}
                            style={{ fontSize: 10, color: C.blue, background: `${C.blue}12`, border: `1px solid ${C.blue}25`, borderRadius: 8, padding: "2px 9px", cursor: "pointer", fontFamily: "inherit", fontWeight: 600 }}>
                            For more →
                        </button>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4, overflowY: "auto", maxHeight: Math.max(260, scoreDist.length * 32) }}>
                        {eligible.map((c, i) => {
                            const score = Math.round((c.finalScore || 0) * 100);
                            const medal = ["🥇", "🥈", "🥉"][i];
                            const barColor = score >= 80 ? C.green : score >= 60 ? C.blue : score >= 40 ? C.teal : C.amber;
                            const RC = {
                                0: { bg: "rgba(255,215,0,.1)", border: "rgba(255,215,0,.35)", text: "#FFD700" },
                                1: { bg: "rgba(192,192,192,.1)", border: "rgba(192,192,192,.35)", text: "#C0C0C0" },
                                2: { bg: "rgba(205,127,50,.1)", border: "rgba(205,127,50,.35)", text: "#CD7F32" },
                            };
                            const rc = RC[i] || { bg: "transparent", border: C.border, text: C.sub };
                            return (
                                <div key={c.id} style={{
                                    display: "flex", alignItems: "center", gap: 10,
                                    padding: "9px 12px", borderRadius: 10,
                                    background: i < 3 ? rc.bg : C.inputBg,
                                    border: `1px solid ${rc.border}`,
                                    width: "100%", boxSizing: "border-box",
                                }}>
                                    {/* Rank badge */}
                                    <div style={{ width: 28, height: 28, borderRadius: 7, display: "flex", alignItems: "center", justifyContent: "center", background: i < 3 ? `${rc.text}18` : C.surface, border: `1px solid ${i < 3 ? rc.border : C.border}`, fontSize: i < 3 ? 15 : 10, fontWeight: 800, color: i < 3 ? rc.text : C.sub, flexShrink: 0 }}>
                                        {medal || `#${i + 1}`}
                                    </div>
                                    {/* Name */}
                                    <div style={{ width: 90, fontSize: 12, color: C.text, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flexShrink: 0 }}>{c.name || "?"}</div>
                                    {/* Progress bar */}
                                    <div style={{ flex: 1, height: 6, background: `${barColor}20`, borderRadius: 3, overflow: "hidden", minWidth: 40 }}>
                                        <div style={{ height: "100%", width: `${score}%`, background: barColor, borderRadius: 3, transition: "width .9s ease", boxShadow: `0 0 6px ${barColor}55` }} />
                                    </div>
                                    {/* Score */}
                                    <div style={{ width: 38, fontSize: 12, fontWeight: 800, color: barColor, textAlign: "right", flexShrink: 0 }}>{score}%</div>
                                    {/* Status dot */}
                                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: c.eligible ? C.green : "#ef4444", flexShrink: 0, boxShadow: `0 0 5px ${c.eligible ? C.green : "#ef4444"}` }} />
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>

            {/* ── Top candidate radar + score cards side by side ─────────── */}
            {top && topRadar.length >= 3 && (
                <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr", gap: 13, alignItems: "stretch" }}>
                    {/* Radar */}
                    <div style={card({ padding: 18 })}>
                        <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 12 }}>Top Candidate — Skill Radar</div>
                        <div style={{ textAlign: "center", marginBottom: 6 }}>
                            <span style={{ fontSize: 14, fontWeight: 800, color: C.blue, background: `${C.blue}12`, padding: "3px 14px", borderRadius: 20, border: `1px solid ${C.blue}30` }}>{top.name}</span>
                        </div>
                        <ResponsiveContainer width="100%" height={240}>
                            <RadarChart data={topRadar} outerRadius="68%" margin={{ top: 10, right: 25, bottom: 10, left: 25 }}>
                                <PolarGrid stroke={C.border} />
                                <PolarAngleAxis dataKey="s" tick={{ fill: C.sub, fontSize: 10 }} />
                                <Radar dataKey="v" stroke={C.chartSkill} fill={C.chartSkill} fillOpacity={0.18} strokeWidth={2.5} dot={{ fill: C.chartSkill, r: 3 }} />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Score breakdown beside radar */}
                    <div style={card({ padding: 18, display: "flex", flexDirection: "column", justifyContent: "space-between" })}>
                        <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 14 }}>Score Breakdown</div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 12, flex: 1, justifyContent: "center" }}>
                            {[
                                { label: "Skill Match", value: Math.round((top.skillScore || 0) * 100), color: C.blue },
                                { label: "Semantic Similarity", value: Math.round((top.semanticScore || 0) * 100), color: C.teal },
                                { label: "Final Score", value: Math.round((top.finalScore || 0) * 100), color: C.amber },
                            ].map(({ label, value, color }) => (
                                <div key={label}>
                                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                                        <span style={{ fontSize: 12, color: C.sub }}>{label}</span>
                                        <span style={{ fontSize: 13, fontWeight: 800, color }}>{value}%</span>
                                    </div>
                                    <div style={{ height: 8, background: `${color}15`, borderRadius: 4, overflow: "hidden" }}>
                                        <div style={{ height: "100%", width: `${value}%`, background: `linear-gradient(90deg,${color},${color}cc)`, borderRadius: 4, boxShadow: `0 0 8px ${color}44`, transition: "width 1s ease" }} />
                                    </div>
                                </div>
                            ))}
                        </div>
                        <div style={{ marginTop: 16, padding: "10px 14px", borderRadius: 10, background: `${C.blue}08`, border: `1px solid ${C.blue}18` }}>
                            <div style={{ fontSize: 11, color: C.sub, marginBottom: 4 }}>Matched Skills ({(top.matched_skills || []).length})</div>
                            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                                {(top.matched_skills || []).map(s => <SkillChip key={s} label={s} matched />)}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* ── Score range distribution — with full explanation ─────────── */}
            <div style={card({ padding: 18 })}>
                <div style={{ marginBottom: 14 }}>
                    <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 4 }}>Score Range Distribution</div>
                    <div style={{ fontSize: 12, color: C.cardText, lineHeight: 1.6 }}>
                        This chart groups all <strong style={{ color: C.text }}>{results.length} screened candidates</strong> into 5 score bands based on their <strong style={{ color: C.amber }}>Final Score</strong> — the weighted combination of skill match and semantic similarity. It helps you quickly see how spread out or concentrated the talent pool is for this role.
                    </div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 8 }}>
                    {[
                        { range: "0–20", label: "Very Low", desc: "Weak match" },
                        { range: "21–40", label: "Below Avg", desc: "Some gaps" },
                        { range: "41–60", label: "Average", desc: "Partial fit" },
                        { range: "61–80", label: "Good", desc: "Strong match" },
                        { range: "81–100", label: "Excellent", desc: "Top talent" },
                    ].map(({ range, label, desc }) => {
                        const count = buckets[range] || 0;
                        const pct = results.length ? Math.round(count / results.length * 100) : 0;
                        const col = range === "81–100" ? "#4aba91" : range === "61–80" ? "#5b9bd5" : range === "41–60" ? "#d4a843" : range === "21–40" ? "#e8854a" : "#e74c5e";
                        const empty = count === 0;
                        return (
                            <div key={range} style={{ textAlign: "center", padding: "14px 8px", borderRadius: 12, background: empty ? `${C.muted}06` : `${col}08`, border: `1px solid ${empty ? C.border : `${col}25`}`, opacity: empty ? 0.4 : 1, transition: "opacity .2s" }}>
                                <div style={{ fontSize: 26, fontWeight: 900, color: empty ? C.muted : col, lineHeight: 1 }}>{count}</div>
                                <div style={{ fontSize: 9, color: empty ? C.muted : col, fontWeight: 700, marginTop: 3, textTransform: "uppercase", letterSpacing: ".05em" }}>{label}</div>
                                <div style={{ fontSize: 10, color: C.sub, margin: "3px 0 1px" }}>{range}</div>
                                <div style={{ fontSize: 10, fontWeight: 700, color: empty ? C.muted : col }}>{pct}%</div>
                                <div style={{ fontSize: 9, color: C.muted, marginTop: 3 }}>{desc}</div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* ── Average scores across eligible ───────────────────────────── */}
            <div style={card({ padding: 18, background: `${C.blue}05`, borderColor: `${C.blue}22` })}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                    <Target size={13} color={C.blue} />
                    <div style={{ fontSize: 13, fontWeight: 700, color: C.blue }}>Average Scores — Shortlisted Candidates</div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "repeat(3,1fr)", gap: 10 }}>
                    {[
                        { label: "Avg Skill Match", value: `${avgSkill}%`, color: C.blue, desc: "Average skill coverage across shortlisted" },
                        { label: "Avg Semantic Similarity", value: `${avgSem}%`, color: C.teal, desc: "Average embedding cosine similarity to JD" },
                        { label: "Avg Final Score", value: `${avgFinal}%`, color: C.amber, desc: "Weighted average using your configured skill/semantic split" },
                    ].map(({ label, value, color, desc }) => (
                        <div key={label} style={{ padding: 14, borderRadius: 10, background: C.surface, border: `1px solid ${C.border}`, textAlign: "center" }}>
                            <div style={{ fontSize: 26, fontWeight: 900, color, marginBottom: 3 }}>{value}</div>
                            <div style={{ fontSize: 12, fontWeight: 700, color: C.text, marginBottom: 4 }}>{label}</div>
                            <div style={{ fontSize: 11, color: C.sub, lineHeight: 1.5 }}>{desc}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* ── Skill Coverage Analysis ────────────────────────────────── */}
            {results.length >= 2 && (
                <SkillCoverageAnalysis results={results} isMobile={isMobile} />
            )}

        </div>
    );
};

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
    const [dark, setDark] = useState(true);
    const [nav, setNav] = useState("upload");
    const [screeningConfig, setScreeningConfig] = useState(null);
    const [results, setResults] = useState([]);          // real API results only
    const [activeModel, setActiveModel] = useState("mpnet");
    const [profileOpen, setProfileOpen] = useState(false);
    // Profile is loaded from localStorage so it survives page reloads
    const [profile, setProfile] = useState(() => {
        try {
            const saved = localStorage.getItem("screeningProfile");
            return saved ? JSON.parse(saved) : { name: "", email: "", role: "", org: "" };
        } catch { return { name: "", email: "", role: "", org: "" }; }
    });
    const [sidebarOpen, setSidebarOpen] = useState(false);

    // Persist profile to localStorage whenever it changes
    const handleSaveProfile = (newProfile) => {
        setProfile(newProfile);
        try { localStorage.setItem("screeningProfile", JSON.stringify(newProfile)); } catch (_) { }
    };

    // Update module-level theme ref on every render
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

    const go = id => { setNav(id); setSidebarOpen(false); };

    // Allow analytics tab to navigate via custom event
    useEffect(() => {
        const handler = e => go(e.detail);
        document.addEventListener("navTo", handler);
        return () => document.removeEventListener("navTo", handler);
    }, []);

    const handleStartScreening = cfg => {
        setScreeningConfig(cfg);
        setNav("processing");
        setSidebarOpen(false);
    };

    const handleProcessingDone = apiResults => {
        setResults(apiResults || []);
        setNav(apiResults?.length ? "dashboard" : "upload");
    };

    // Switch model — updates state and tells Flask
    const handleModelChange = async key => {
        setActiveModel(key);
        try {
            await fetch(`${BASE}/api/model`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: key }),
            });
        } catch (_) { }
    };

    // Map nav keys to view components
    const viewMap = {
        dashboard: <DashboardView results={results} onNav={go} isMobile={isMobile} activeModel={activeModel} onModelChange={handleModelChange} />,
        upload: <UploadView onStartScreening={handleStartScreening} activeModel={activeModel} onModelChange={handleModelChange} isMobile={isMobile} />,
        processing: <ProcessingView config={screeningConfig} onDone={handleProcessingDone} />,
        config: <JobConfigView />,
        candidates: <CandidatesView results={results} onNav={go} isMobile={isMobile} />,
        analytics: <AnalyticsView results={results} isMobile={isMobile} />,
    };

    // Sidebar content (shared between desktop sticky + mobile drawer)
    const SidebarContent = () => (
        <>
            {/* Logo — same height as topbar */}
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

            {/* Pipeline status */}
            <div style={{ padding: "8px 10px", borderRadius: 8, background: `${C.green}08`, border: `1px solid ${C.green}20`, fontSize: 11, color: C.green, display: "flex", alignItems: "center", gap: 7 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: C.green, animation: "pulse 2s infinite", flexShrink: 0 }} />
                {results.length > 0 ? `${results.length} candidates ranked` : "Pipeline Ready"}
            </div>
        </>
    );

    return (
        <>
            {/* ── Global CSS ─────────────────────────────────────────────── */}
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');
        *,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
        html { font-size:16px; }
        body { background:${C.bg}; font-family:'Plus Jakarta Sans',sans-serif; color:${C.text}; -webkit-font-smoothing:antialiased; font-weight:500; }
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

        /* Profile modal — spring pop in, GPU accelerated */
        @keyframes modalIn {
          from { opacity:0; transform:translate3d(-50%,-48%,0) scale(.97) }
          to   { opacity:1; transform:translate3d(-50%,-50%,0) scale(1) }
        }

        /* Range sliders */
        input[type=range] { -webkit-appearance:none; appearance:none; height:4px; background:${C.inputBg}; border-radius:2px; width:100%; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; width:14px; height:14px; border-radius:50%; background:currentColor; cursor:pointer; }
        textarea { color-scheme:${dark ? "dark" : "light"}; }
        button,input,textarea,select { font-family:'Plus Jakarta Sans',sans-serif; }
      `}</style>

            {/* ── Profile modal ──────────────────────────────────────────── */}
            {profileOpen && <ProfileModal profile={profile} onSave={handleSaveProfile} onClose={() => setProfileOpen(false)} />}

            {/* ── Mobile sidebar drawer ──────────────────────────────────── */}
            {isMobile && sidebarOpen && (
                <>
                    <div onClick={() => setSidebarOpen(false)} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,.55)", backdropFilter: "blur(4px)", zIndex: 300 }} />
                    <div style={{ position: "fixed", left: 0, top: 0, bottom: 0, width: 210, background: C.sidebarBg, borderRight: `1px solid ${C.border}`, zIndex: 310, padding: "20px 11px", display: "flex", flexDirection: "column", animation: "slideInLeft .24s ease" }}>
                        <SidebarContent />
                    </div>
                </>
            )}

            {/* ── Layout ─────────────────────────────────────────────────── */}
            <div style={{ display: "flex", minHeight: "100vh", background: C.bg, color: C.text }}>

                {/* Desktop sidebar */}
                {!isMobile && (
                    <aside style={{ width: isTablet ? 200 : 218, flexShrink: 0, borderRight: `1px solid ${C.border}`, display: "flex", flexDirection: "column", padding: "20px 11px", position: "sticky", top: 0, height: "100vh", background: C.sidebarBg }}>
                        <SidebarContent />
                    </aside>
                )}

                {/* Main content area */}
                <main style={{ flex: 1, display: "flex", flexDirection: "column", overflowY: "auto", maxHeight: "100vh", minWidth: 0 }}>

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
                    <div style={{ flex: 1, padding: isMobile ? "16px 13px" : "22px 26px", animation: "fadeIn .18s ease" }} key={nav}>
                        {viewMap[nav]}
                    </div>
                </main>
            </div>
        </>
    );
}