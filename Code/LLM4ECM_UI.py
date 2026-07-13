import html
import io
import markdown
import re
import socket
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from langchain_core.messages import HumanMessage

from LLM4ECM import (
    DEFAULT_MODEL,
    convert_nondeterministic_to_deterministic,
    update_aml_model,
    build_main_agent,
    get_last_nondeterministic_submodel_json,
    get_nondeterministic_agent_call_count,
    get_last_deterministic_submodel_json,
    get_deterministic_agent_call_count,
    reset_latest_nondeterministic_submodel_json,
    reset_latest_deterministic_submodel_json,
    reset_latest_submodel_jsons,
    reset_agent_short_memory,
    set_aml_internal_elements,
)




def find_free_port(start_port=8094, host="127.0.0.1"):
    """Return the first available local port at or above start_port."""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port

    raise RuntimeError("No free local port found.")


def get_ollama_models():
    """Return model names from `ollama list`."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return [DEFAULT_MODEL]

    models = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])

    return models or [DEFAULT_MODEL]


def get_running_ollama_models():
    """Return model names that Ollama reports as currently loaded."""
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return set()

    running = set()
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            running.add(parts[0])

    return running


def parse_internal_element_records(xml_source):
    """Load InternalElement names and IDs from an AML/XML/SML file-like object or path."""
    try:
        tree = ET.parse(xml_source)
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse AML/XML/SML file: {exc}") from exc

    root = tree.getroot()
    tag = "InternalElement"
    if root.tag.startswith("{"):
        namespace_uri = root.tag[1:root.tag.index("}")]
        tag = f"{{{namespace_uri}}}InternalElement"

    records = []
    for elem in root.iter(tag):
        name = elem.attrib.get("Name", "").strip()
        if not name or name == "InternalElement":
            continue

        element_id = (
            elem.attrib.get("ID")
            or elem.attrib.get("Id")
            or elem.attrib.get("id")
            or name
        )
        records.append({"name": name, "id": element_id.strip()})

    return records


def parse_internal_element_names(xml_source):
    """Load InternalElement names from an AML/XML/SML file-like object or path."""
    return [record["name"] for record in parse_internal_element_records(xml_source)]


def get_internal_element_names():
    """Load InternalElement names from Vera.xml."""
    xml_path = Path(__file__).parent / "Vera.xml"
    if not xml_path.exists():
        return []

    try:
        records = parse_internal_element_records(xml_path)
    except ValueError:
        return []

    set_aml_internal_elements(records)
    return [record["name"] for record in records]


# ============================================================
# FLASK UI
# ============================================================

app = Flask(__name__)
agent_short_memory_reset_done = False

HTML_PAGE = r"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM4ECM</title>

    <style>
        * {
            box-sizing: border-box;
        }

        :root {
            --ink: #172033;
            --muted: #667085;
            --panel: rgba(255, 252, 246, 0.94);
            --panel-strong: #fffaf1;
            --line: rgba(91, 71, 46, 0.16);
            --line-strong: #e8d6b7;
            --warm: #fff7ed;
            --mint: #eefcf8;
            --teal: #0f766e;
            --amber: #f97316;
            --amber-dark: #c2410c;
            --violet: #5b4b8a;
            --shadow: 0 24px 70px rgba(18, 27, 43, 0.28);
            --soft-shadow: 0 10px 28px rgba(55, 43, 27, 0.12);
        }

        body {
            color: var(--ink);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            margin: 0;
            min-height: 100vh;
            background:
                linear-gradient(135deg, rgba(12, 74, 110, 0.88) 0%, rgba(17, 24, 39, 0.94) 42%, rgba(67, 56, 202, 0.82) 100%),
                repeating-linear-gradient(90deg, rgba(255,255,255,0.06) 0 1px, transparent 1px 56px),
                repeating-linear-gradient(0deg, rgba(255,255,255,0.045) 0 1px, transparent 1px 56px);
        }

        .app {
            height: 100vh;
            min-height: 0;
            display: flex;
            flex-direction: column;
            padding: 16px;
            gap: 14px;
            overflow: hidden;
        }

        .title-bar {
            flex: 0 0 auto;
            position: relative;
            background: linear-gradient(90deg, rgba(255, 247, 237, 0.97) 0%, rgba(239, 252, 248, 0.96) 56%, rgba(243, 240, 255, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.68);
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: var(--shadow);
            overflow: hidden;
        }

        .title-bar::before {
            content: "";
            position: absolute;
            inset: 0;
            border-top: 3px solid rgba(249, 115, 22, 0.76);
            pointer-events: none;
        }

        h1 {
            color: #1e293b;
            font-size: 24px;
            line-height: 1;
            margin: 0;
            letter-spacing: 0;
            text-align: center;
        }

        .github-link {
            position: absolute;
            top: 50%;
            right: 14px;
            z-index: 1;
            width: 38px;
            height: 38px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(232, 214, 183, 0.95);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.78);
            color: #25324a;
            box-shadow: 0 8px 18px rgba(55, 43, 27, 0.12);
            transform: translateY(-50%);
            transition: transform 140ms ease, background 140ms ease, box-shadow 140ms ease;
        }

        .github-link:hover {
            background: #ffffff;
            box-shadow: 0 12px 24px rgba(55, 43, 27, 0.18);
            transform: translateY(-52%);
        }

        .github-link svg {
            width: 22px;
            height: 22px;
            fill: currentColor;
        }

        .workspace {
            flex: 1 1 auto;
            display: grid;
            grid-template-columns: minmax(285px, 0.92fr) minmax(300px, 1.04fr) minmax(300px, 1.04fr);
            gap: 14px;
            min-height: 0;
            overflow: hidden;
        }

        .panel {
            min-width: 0;
            min-height: 0;
            background: rgba(255, 250, 241, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.64);
            border-radius: 8px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
        }

        .left-panel {
            --chat-box-height: 220px;
            display: grid;
            grid-template-rows: auto minmax(160px, 1fr) minmax(160px, 1fr);
            gap: 12px;
            padding: 12px;
        }

        .stack-panel {
            display: grid;
            grid-template-rows: repeat(2, minmax(0, 1fr));
            gap: 12px;
            padding: 12px;
        }


        .reset-button {
            width: auto;
            min-width: 94px;
            height: 34px;
            padding: 0 12px;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            background: linear-gradient(180deg, #ffffff 0%, #fff7ed 100%);
            color: #9a3412;
            font-size: 13px;
            font-weight: 800;
            box-shadow: 0 7px 16px rgba(154, 52, 18, 0.12);
        }

        .aml-panel {
            grid-template-rows: 62px minmax(0, 1fr);
            transition: grid-template-rows 180ms ease;
        }

        .aml-panel.elements-open {
            grid-template-rows: minmax(205px, 0.58fr) minmax(0, 1fr);
        }

        .content-box,
        .model-card,
        .prompt-card,
        #answer-box {
            border: 1px solid var(--line-strong);
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(255, 249, 237, 0.97) 100%);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.94), var(--soft-shadow);
            overflow: hidden;
        }

        .content-box {
            min-height: 0;
            display: flex;
            flex-direction: column;
        }

        .internal-elements-box.collapsed .box-body {
            display: none;
        }

        .box-title,
        .field-label {
            margin: 0;
            color: #25324a;
            font-size: 14px;
            font-weight: 800;
            letter-spacing: 0;
        }

        .box-title {
            min-height: 42px;
            display: flex;
            align-items: center;
            padding: 10px 12px;
            border-bottom: 1px solid var(--line-strong);
            background: linear-gradient(90deg, var(--warm) 0%, var(--mint) 68%, #f3f0ff 100%);
        }

        .box-title::before,
        .field-label::before {
            content: "";
            width: 7px;
            height: 7px;
            flex: 0 0 auto;
            margin-right: 8px;
            border-radius: 999px;
            background: var(--amber);
            box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.13);
        }

        .box-header,
        .model-card-header,
        .prompt-card-header {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 10px;
            align-items: center;
            min-height: 60px;
            padding: 10px 12px;
            border-bottom: 1px solid var(--line-strong);
            background: linear-gradient(90deg, var(--warm) 0%, var(--mint) 68%, #f3f0ff 100%);
        }

        .box-header .box-title {
            min-height: 0;
            padding: 0;
            border-bottom: 0;
            background: transparent;
        }

        .title-meta,
        .model-hint,
        .prompt-hint {
            display: block;
            color: #7a6b54;
            font-size: 12px;
            font-weight: 700;
            margin-top: 3px;
            white-space: nowrap;
        }

        .box-body {
            flex: 1 1 auto;
            min-height: 0;
            overflow-y: auto;
            overflow-x: auto;
            padding: 12px;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.56), rgba(255,250,241,0.7)),
                repeating-linear-gradient(135deg, rgba(15,118,110,0.035) 0 1px, transparent 1px 12px);
        }

        #aml-source-output {
            margin: 0;
            color: #1f2937;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 12.5px;
            line-height: 1.5;
            tab-size: 2;
            white-space: pre;
        }

        #aml-source-output .xml-tag {
            color: #0f766e;
            font-weight: 800;
        }

        #aml-source-output .xml-attr {
            color: #7c3aed;
        }

        #aml-source-output .xml-value {
            color: #c2410c;
        }

        #aml-source-output .xml-comment {
            color: #64748b;
            font-style: italic;
        }

        #aml-source-output .xml-update-marker {
            background: #fef3c7;
            color: #92400e;
            border-radius: 4px;
            padding: 1px 3px;
            font-weight: 800;
        }

        #aml-source-output .xml-declaration {
            color: #0369a1;
            font-weight: 750;
        }

        #nondeterministic-output .json-key {
            color: #7c3aed;
            font-weight: 600;
        }

        #nondeterministic-output .json-string {
            color: #c2410c;
        }

        #nondeterministic-output .json-number {
            color: #0369a1;
        }

        #nondeterministic-output .json-boolean {
            color: #059669;
            font-weight: 600;
        }

        #nondeterministic-output .json-null {
            color: #64748b;
            font-weight: 600;
        }

        #deterministic-output .json-key {
            color: #7c3aed;
            font-weight: 600;
        }

        #deterministic-output .json-string {
            color: #c2410c;
        }

        #deterministic-output .json-number {
            color: #0369a1;
        }

        #deterministic-output .json-boolean {
            color: #059669;
            font-weight: 600;
        }

        #deterministic-output .json-null {
            color: #64748b;
            font-weight: 600;
        }

        .content-box,
        #answer-box,
        #message {
            scrollbar-color: #b98553 #fff1dd;
            scrollbar-width: thin;
        }

        .content-box::-webkit-scrollbar,
        #answer-box::-webkit-scrollbar,
        #message::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        .content-box::-webkit-scrollbar-track,
        #answer-box::-webkit-scrollbar-track,
        #message::-webkit-scrollbar-track {
            background: #fff1dd;
        }

        .content-box::-webkit-scrollbar-thumb,
        #answer-box::-webkit-scrollbar-thumb,
        #message::-webkit-scrollbar-thumb {
            background: #b98553;
            border: 2px solid #fff1dd;
            border-radius: 999px;
        }

        .model-card-header,
        .prompt-card-header {
            min-height: 48px;
        }

        .model-card .field-label,
        .prompt-card .field-label {
            display: flex;
            align-items: center;
            margin-bottom: 0;
        }

        .model-card-body,
        .prompt-card-body {
            padding: 12px;
        }

        .model-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 10px;
            align-items: center;
        }

        #model-select,
        #message {
            width: 100%;
            border: 1px solid #e9d8bc;
            border-radius: 8px;
            background: #ffffff;
            color: var(--ink);
            font-size: 15px;
            outline: none;
        }

        #model-select:focus,
        #message:focus {
            border-color: var(--amber);
            box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.16);
        }

        #model-select {
            height: 40px;
            padding: 0 34px 0 12px;
            color: #233146;
            font-weight: 800;
            box-shadow: 0 6px 16px rgba(99, 72, 35, 0.08);
        }

        #model-status {
            min-width: 88px;
            height: 40px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            background: #ffffff;
            color: #9a3412;
            font-size: 13px;
            font-weight: 800;
            white-space: nowrap;
            box-shadow: 0 6px 16px rgba(154, 52, 18, 0.09);
        }

        #model-status.running {
            background: #e7fbf5;
            border-color: #99e6d2;
            color: var(--teal);
        }

        #model-status.active {
            background: #fff3df;
            border-color: #fdba74;
            color: var(--amber-dark);
        }

        #model-status.available {
            color: #7a5b2f;
        }

        #answer-box {
            min-height: 0;
            overflow-y: auto;
            overflow-x: auto;
            padding: 14px;
            line-height: 1.5;
            color: var(--ink);
            background:
                linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,250,241,0.94)),
                repeating-linear-gradient(135deg, rgba(91,75,138,0.04) 0 1px, transparent 1px 14px);
        }

        #answer-box.empty {
            color: #7a6b54;
            font-weight: 700;
        }

        .prompt-card {
            min-height: 0;
            display: grid;
            grid-template-rows: auto minmax(0, 1fr);
        }

        .prompt-card-body {
            min-height: 0;
            display: grid;
            grid-template-rows: minmax(0, 1fr) auto;
            gap: 10px;
        }

        #message {
            height: 100%;
            min-height: 0;
            max-height: none;
            resize: none;
            overflow-y: auto;
            overflow-x: auto;
            padding: 12px;
            line-height: 1.45;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
        }

        button {
            width: 100%;
            height: 40px;
            margin-top: 0;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            background: linear-gradient(135deg, #f97316 0%, #d9480f 100%);
            color: #ffffff;
            font-size: 15px;
            font-weight: 800;
            box-shadow: 0 12px 24px rgba(249, 115, 22, 0.28);
            transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease;
        }

        button:hover {
            filter: brightness(1.03);
            transform: translateY(-1px);
            box-shadow: 0 16px 28px rgba(249, 115, 22, 0.32);
        }

        button:active {
            transform: translateY(0);
        }

        button:disabled {
            cursor: wait;
            background: #fb923c;
            transform: none;
        }

        .box-actions {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .import-button,
        .export-button,
        .update-button,
        .toggle-button {
            width: auto;
            min-width: 86px;
            height: 36px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 7px;
            padding: 0 12px;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            background: linear-gradient(180deg, #ffffff 0%, #fff7ed 100%);
            color: #9a3412;
            box-shadow: 0 7px 16px rgba(154, 52, 18, 0.12);
            font-size: 13px;
            font-weight: 800;
            line-height: 1;
            transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease;
        }

        .import-button:hover,
        .export-button:hover,
        .update-button:hover,
        .toggle-button:hover {
            background: linear-gradient(180deg, #fffaf4 0%, #ffedd5 100%);
            transform: translateY(-1px);
            box-shadow: 0 10px 20px rgba(154, 52, 18, 0.16);
        }

        .import-button:active,
        .export-button:active,
        .update-button:active,
        .toggle-button:active {
            transform: translateY(0);
            box-shadow: 0 6px 14px rgba(154, 52, 18, 0.12);
        }

        .import-button {
            position: relative;
            overflow: hidden;
        }

        .import-button input {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }

        .import-icon,
        .export-icon,
        .update-icon,
        .toggle-icon {
            width: 16px;
            height: 16px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: #ffedd5;
            color: #c2410c;
            font-size: 13px;
            line-height: 1;
        }

        .element-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(112px, 1fr));
            gap: 6px;
            margin: 0;
            padding: 0;
            list-style: none;
        }

        .element-list li {
            min-width: 0;
            border: 1px solid #ead8b8;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.92);
            color: #263244;
            font-size: 10.5px;
            font-weight: 750;
            padding: 6px 7px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            box-shadow: 0 6px 16px rgba(99, 72, 35, 0.07);
        }

        em {
            color: #7a6b54;
            font-weight: 700;
        }

        @media (max-width: 1100px) {
            .workspace {
                grid-template-columns: 1fr 1fr;
            }

            .left-panel {
                grid-column: 1 / -1;
                grid-template-columns: minmax(250px, 0.8fr) minmax(300px, 1fr) minmax(280px, 0.9fr);
                grid-template-rows: minmax(0, 1fr);
            }
        }

        @media (max-width: 900px) {
            .app {
                height: auto;
                min-height: 100vh;
                overflow: visible;
            }

            .workspace,
            .left-panel {
                grid-template-columns: 1fr;
                grid-template-rows: auto;
                overflow: visible;
            }

            .panel,
            .stack-panel {
                min-height: 360px;
            }

            .left-panel {
                min-height: auto;
            }

            #answer-box {
                min-height: 180px;
            }
        }
    </style>
</head>

<body>
<div class="app">
    <header class="title-bar">
        <h1>LLM4ECM</h1>
        <a
            class="github-link"
            href="https://github.com/HRAhvanouee/LLM4ECM"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Open LLM4ECM on GitHub"
            title="GitHub: HRAhvanouee/LLM4ECM"
        >
            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <path d="M12 2C6.48 2 2 6.59 2 12.25c0 4.53 2.87 8.37 6.84 9.73.5.09.68-.22.68-.49 0-.24-.01-.88-.01-1.73-2.78.62-3.37-1.37-3.37-1.37-.45-1.18-1.11-1.49-1.11-1.49-.91-.64.07-.63.07-.63 1 .07 1.53 1.06 1.53 1.06.89 1.56 2.34 1.11 2.91.85.09-.66.35-1.11.63-1.37-2.22-.26-4.56-1.14-4.56-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.05A9.3 9.3 0 0 1 12 7c.85 0 1.71.12 2.51.34 1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.81-4.57 5.07.36.32.68.94.68 1.9 0 1.37-.01 2.47-.01 2.81 0 .27.18.59.69.49A10.14 10.14 0 0 0 22 12.25C22 6.59 17.52 2 12 2Z"/>
            </svg>
        </a>
    </header>

    <main class="workspace">
        <section class="panel left-panel">
            <div class="model-card">
                <div class="model-card-header">
                    <label class="field-label" for="model-select">AI Model</label>
                    <span class="model-hint">local runtime</span>
                </div>
                <div class="model-card-body">
                    <div class="model-row">
                        <select id="model-select">
                            {% for model in models %}
                            <option value="{{ model }}" {% if model == default_model %}selected{% endif %}>{{ model }}</option>
                            {% endfor %}
                        </select>
                        <span id="model-status" class="{{ default_status_class }}">{{ default_status }}</span>
                    </div>
                </div>
            </div>

            <div id="answer-box" class="empty">AI answer will appear here.</div>

            <div class="prompt-card">
                <div class="prompt-card-header">
                    <label class="field-label" for="message">User message</label>
                    <span class="prompt-hint">compose</span>
                </div>
                <div class="prompt-card-body">
                    <textarea id="message" placeholder="Type your message...">I have engineering feedback, and I want to convert this itemized feedback into the Nondeterministic Engineering Change format. Please adhere strictly to the JSON structure of the submodel and the JSON format of each record and item when filling the submodel. I need the answer only in valid, well-structured JSON format.

1-  Add a new Attribute "Pressure" to Tank1_T1 and update the pressure of Tank1_T1 to 5 kPa.
2 - Add a new Attribute "Level" to Tank2_T2.
3 - Update the rotation speed of PR_Pumpe from 25 RPM to 30 RPM.
4 - Delete the K1 Tank.</textarea>
                    <button id="send-button" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </section>

        <section class="panel stack-panel output-panel">
            <div class="content-box">
                <div class="box-header">
                    <h2 class="box-title">Nondeterministic Changes Submodel</h2>
                    <button id="reset-nondeterministic" class="reset-button" type="button" onclick="resetNondeterministicBox()">Reset</button>
                </div>
                <div class="box-body"><pre id="nondeterministic-output">{{ nondeterministic_submodel_json }}</pre></div>
            </div>
            <div class="content-box">
                <div class="box-header">
                    <h2 class="box-title">Deterministic Changes Submodel</h2>
                    <button id="reset-deterministic" class="reset-button" type="button" onclick="resetDeterministicBox()">Reset</button>
                </div>
                <div class="box-body"><pre id="deterministic-output">{{ deterministic_submodel_json }}</pre></div>
            </div>
        </section>

        <section id="aml-panel" class="panel stack-panel aml-panel">
            <div id="internal-elements-box" class="content-box internal-elements-box collapsed">
                <div class="box-header">
                    <div>
                        <h2 class="box-title">Internal Elements of AML Model</h2>
                        <span class="title-meta">{{ internal_elements|length }} elements from Vera.xml</span>
                    </div>
                    <div class="box-actions">
                        <label class="import-button" title="Import AML, XML, or SML file">
                            <span class="import-icon" aria-hidden="true">↑</span>
                            <span>Import</span>
                            <input id="aml-import" type="file" accept=".aml,.xml,.sml,text/xml,application/xml">
                        </label>
                        <button
                            id="toggle-elements"
                            class="toggle-button"
                            type="button"
                            aria-expanded="false"
                            aria-controls="internal-elements-body"
                            onclick="toggleInternalElements()"
                        >
                            <span class="toggle-icon" aria-hidden="true">+</span>
                            <span class="toggle-text">Show</span>
                        </button>
                    </div>
                </div>
                <div id="internal-elements-body" class="box-body">
                    {% if internal_elements %}
                        <ul class="element-list">
                            {% for name in internal_elements %}
                                <li>{{ name }}</li>
                            {% endfor %}
                        </ul>
                    {% else %}
                        <em>No internal elements loaded from Vera.xml.</em>
                    {% endif %}
                </div>
            </div>
            <div class="content-box">
                <div class="box-header">
                    <h2 class="box-title">Updated AML Model</h2>
                    <div class="box-actions">
                        <button id="update-aml" class="update-button" type="button" onclick="applyAmlUpdates()">
                            <span class="update-icon" aria-hidden="true">↻</span>
                            <span>Update</span>
                        </button>
                        <button id="export-aml" class="export-button" type="button" onclick="exportUpdatedAml()">
                            <span class="export-icon" aria-hidden="true">↓</span>
                            <span>Export</span>
                        </button>
                    </div>
                </div>
                <div class="box-body"><pre id="aml-source-output" data-source-name=""><em>Imported AML/XML code will appear here.</em></pre></div>
            </div>
        </section>
    </main>
</div>

<script>
function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
}

function highlightXml(source) {
    return escapeHtml(source).replace(
        /(&lt;!--[\s\S]*?--&gt;)|(&lt;\?[\s\S]*?\?&gt;)|(&lt;\/?[\w:.-]+)([\s\S]*?)(\/?&gt;)/g,
        function(match, comment, declaration, tagStart, attributes, tagEnd) {
            if (comment) {
                return `<span class="xml-comment">${comment}</span>`;
            }

            if (declaration) {
                return `<span class="xml-declaration">${declaration}</span>`;
            }

            const highlightedAttributes = attributes.replace(
                /([\w:.-]+)(=)("[^"]*"|'[^']*')/g,
                '<span class="xml-attr">$1</span>$2<span class="xml-value">$3</span>'
            );

            return `<span class="xml-tag">${tagStart}</span>${highlightedAttributes}<span class="xml-tag">${tagEnd}</span>`;
        }
    );
}

function highlightAmlUpdateMarkers(source) {
    return source.replace(
        /(<span class="xml-attr">LLM4ECM(?:Change|Action)<\/span>=<span class="xml-value">[^<]+<\/span>)/g,
        '<span class="xml-update-marker">$1</span>'
    );
}

function setHighlightedXml(element, source) {
    element.innerHTML = source
        ? highlightAmlUpdateMarkers(highlightXml(source))
        : "<em>Imported AML/XML code will appear here.</em>";
}

function highlightJson(source) {
    const escaped = escapeHtml(source);
    return escaped.replace(
        /("(?:\\.|[^\\"])*")\s*:|"(?:\\.|[^\\"])*"|\b(true|false|null)\b|\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/g,
        function(match) {
            if (match.includes(':')) {
                const key = match.match(/"[^"]*"/)[0];
                return `<span class="json-key">${key}</span>:`;
            }
            if (match.startsWith('"')) {
                return `<span class="json-string">${match}</span>`;
            }
            if (match === 'true' || match === 'false') {
                return `<span class="json-boolean">${match}</span>`;
            }
            if (match === 'null') {
                return `<span class="json-null">${match}</span>`;
            }
            return `<span class="json-number">${match}</span>`;
        }
    );
}

function setHighlightedJson(element, source) {
    element.innerHTML = source
        ? highlightJson(source)
        : "";
}

function getExportFileName(sourceName) {
    if (!sourceName) {
        return "updated-aml-model.xml";
    }

    const lastDot = sourceName.lastIndexOf(".");
    const baseName = lastDot > 0 ? sourceName.slice(0, lastDot) : sourceName;
    const extension = lastDot > 0 ? sourceName.slice(lastDot) : ".xml";
    return `${baseName}-updated${extension}`;
}

function exportUpdatedAml() {
    const amlSourceOutput = document.getElementById("aml-source-output");
    const sourceText = amlSourceOutput.textContent.trim();

    if (!sourceText || !amlSourceOutput.dataset.sourceName) {
        return;
    }

    const blob = new Blob([sourceText], {type: "application/xml;charset=utf-8"});
    const downloadUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = getExportFileName(amlSourceOutput.dataset.sourceName);
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(downloadUrl);
}

async function applyAmlUpdates() {
    const amlSourceOutput = document.getElementById("aml-source-output");
    const deterministicOutput = document.getElementById("deterministic-output");
    const modelSelect = document.getElementById("model-select");
    const updateButton = document.getElementById("update-aml");
    const amlModel = amlSourceOutput.textContent.trim();
    const deterministicContext = deterministicOutput.textContent.trim();

    if (!amlModel || !amlSourceOutput.dataset.sourceName || !deterministicContext) {
        return;
    }

    updateButton.disabled = true;
    updateButton.querySelector("span:last-child").textContent = "Updating";

    try {
        const response = await fetch("/update-aml", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                aml_model: amlModel,
                deterministic_context: deterministicContext,
                model: modelSelect.value
            })
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Could not update AML model.");
        }

        setHighlightedXml(amlSourceOutput, data.updated_aml_model || "");
        amlSourceOutput.dataset.sourceName = data.source_name || amlSourceOutput.dataset.sourceName;
    } catch (error) {
        amlSourceOutput.innerHTML = `<em>${escapeHtml(error.message)}</em>`;
    } finally {
        updateButton.disabled = false;
        updateButton.querySelector("span:last-child").textContent = "Update";
    }
}

function toggleInternalElements() {
    const amlPanel = document.getElementById("aml-panel");
    const elementsBox = document.getElementById("internal-elements-box");
    const toggleButton = document.getElementById("toggle-elements");
    const toggleIcon = toggleButton.querySelector(".toggle-icon");
    const toggleText = toggleButton.querySelector(".toggle-text");
    const isOpening = elementsBox.classList.contains("collapsed");

    amlPanel.classList.toggle("elements-open", isOpening);
    elementsBox.classList.toggle("collapsed", !isOpening);
    toggleButton.setAttribute("aria-expanded", String(isOpening));
    toggleIcon.textContent = isOpening ? "-" : "+";
    toggleText.textContent = isOpening ? "Hide" : "Show";
}

async function refreshModelStatus() {
    const modelSelect = document.getElementById("model-select");
    const status = document.getElementById("model-status");

    status.textContent = "Checking";
    status.className = "";

    try {
        const response = await fetch(`/model-status?model=${encodeURIComponent(modelSelect.value)}`);
        const data = await response.json();
        status.textContent = data.status;
        status.className = data.status_class;
    } catch (error) {
        status.textContent = "Unknown";
        status.className = "available";
    }
}

async function sendMessage() {
    const input = document.getElementById("message");
    const answerBox = document.getElementById("answer-box");
    const nondeterministicOutput = document.getElementById("nondeterministic-output");
    const deterministicOutput = document.getElementById("deterministic-output");
    const amlSourceOutput = document.getElementById("aml-source-output");
    const modelSelect = document.getElementById("model-select");
    const sendButton = document.getElementById("send-button");
    const message = input.value.trim();

    if (!message) {
        return;
    }

    const status = document.getElementById("model-status");

    sendButton.disabled = true;
    status.textContent = "Active";
    status.className = "active";
    answerBox.classList.remove("empty");
    answerBox.innerHTML = "Thinking...";

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                model: modelSelect.value,
                nondeterministic_context: nondeterministicOutput.textContent.trim(),
                deterministic_context: deterministicOutput.textContent.trim(),
                aml_model: amlSourceOutput.textContent.trim(),
                aml_source_name: amlSourceOutput.dataset.sourceName || ""
            })
        });

        const data = await response.json();
        answerBox.innerHTML = data.reply;
        if (data.nondeterministic_submodel_json) {
            setHighlightedJson(nondeterministicOutput, data.nondeterministic_submodel_json);
        }
        if (data.deterministic_submodel_json) {
            setHighlightedJson(deterministicOutput, data.deterministic_submodel_json);
        }
        if (data.updated_aml_model) {
            setHighlightedXml(amlSourceOutput, data.updated_aml_model);
            amlSourceOutput.dataset.sourceName = data.aml_source_name || amlSourceOutput.dataset.sourceName;
        }
        input.value = "";
    } catch (error) {
        answerBox.innerHTML = `<pre>Error: ${error}</pre>`;
    } finally {
        sendButton.disabled = false;
        refreshModelStatus();
        input.focus();
    }
}


async function resetNondeterministicBox() {
    document.getElementById("nondeterministic-output").textContent = "";

    try {
        await fetch("/reset-nondeterministic", {method: "POST"});
    } catch (error) {
        // The visual reset is still useful even if the server reset fails.
    }
}

async function resetDeterministicBox() {
    document.getElementById("deterministic-output").textContent = "";

    try {
        await fetch("/reset-deterministic", {method: "POST"});
    } catch (error) {
        // The visual reset is still useful even if the server reset fails.
    }
}

async function importInternalElements(event) {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    const body = new FormData();
    body.append("aml_file", file);

    try {
        const response = await fetch("/internal-elements", {
            method: "POST",
            body: body
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Could not import file.");
        }

        const meta = document.querySelector("#internal-elements-box .title-meta");
        const bodyBox = document.getElementById("internal-elements-body");
        const amlSourceOutput = document.getElementById("aml-source-output");
        meta.textContent = data.internal_elements.length + " elements from " + data.source_name;
        setHighlightedXml(amlSourceOutput, data.aml_source || "");
        amlSourceOutput.dataset.sourceName = data.source_name || "";

        if (!data.internal_elements.length) {
            bodyBox.innerHTML = "<em>No internal elements found in imported file.</em>";
        } else {
            bodyBox.innerHTML = `<ul class="element-list">${data.internal_elements
                .map((name) => `<li title="${name.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')}">${name.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')}</li>`)
                .join("")}</ul>`;
        }

        const elementsBox = document.getElementById("internal-elements-box");
        if (elementsBox.classList.contains("collapsed")) {
            toggleInternalElements();
        }
    } catch (error) {
        const bodyBox = document.getElementById("internal-elements-body");
        bodyBox.innerHTML = `<em>${error.message}</em>`;
        if (document.getElementById("internal-elements-box").classList.contains("collapsed")) {
            toggleInternalElements();
        }
    } finally {
        event.target.value = "";
    }
}

function initJsonHighlighting() {
    const nondeterministicOutput = document.getElementById("nondeterministic-output");
    const deterministicOutput = document.getElementById("deterministic-output");

    if (nondeterministicOutput && nondeterministicOutput.textContent.trim()) {
        setHighlightedJson(nondeterministicOutput, nondeterministicOutput.textContent);
    }

    if (deterministicOutput && deterministicOutput.textContent.trim()) {
        setHighlightedJson(deterministicOutput, deterministicOutput.textContent);
    }
}

document.getElementById("aml-import").addEventListener("change", importInternalElements);

document.getElementById("model-select").addEventListener("change", refreshModelStatus);

document.addEventListener("DOMContentLoaded", initJsonHighlighting);
</script>
</body>
</html>
"""

@app.route("/")
def home():
    models = get_ollama_models()
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]
    running_models = get_running_ollama_models()
    default_status = "Running" if default_model in running_models else "Available"
    default_status_class = default_status.lower()
    internal_elements = get_internal_element_names()
    nondeterministic_submodel_json = get_last_nondeterministic_submodel_json()
    deterministic_submodel_json = get_last_deterministic_submodel_json()
    return render_template_string(
        HTML_PAGE,
        models=models,
        default_model=default_model,
        default_status=default_status,
        default_status_class=default_status_class,
        internal_elements=internal_elements,
        nondeterministic_submodel_json=nondeterministic_submodel_json,
        deterministic_submodel_json=deterministic_submodel_json,
    )


@app.route("/internal-elements", methods=["POST"])
def internal_elements_from_file():
    uploaded_file = request.files.get("aml_file")
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"error": "Please select an AML, XML, or SML file."}), 400

    uploaded_bytes = uploaded_file.read()
    aml_source = uploaded_bytes.decode("utf-8", errors="replace")

    try:
        internal_element_records = parse_internal_element_records(io.BytesIO(uploaded_bytes))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    set_aml_internal_elements(internal_element_records)
    internal_elements = [record["name"] for record in internal_element_records]

    return jsonify({
        "internal_elements": internal_elements,
        "source_name": Path(uploaded_file.filename).name,
        "aml_source": aml_source,
    })


@app.route("/model-status")
def model_status():
    model_name = request.args.get("model") or DEFAULT_MODEL
    available_models = get_ollama_models()

    if model_name not in available_models:
        return jsonify({
            "status": "Missing",
            "status_class": "available"
        })

    running_models = get_running_ollama_models()
    status = "Running" if model_name in running_models else "Available"

    return jsonify({
        "status": status,
        "status_class": status.lower()
    })



@app.route("/reset-nondeterministic", methods=["POST"])
def reset_nondeterministic():
    reset_latest_nondeterministic_submodel_json()
    return jsonify({"ok": True})


@app.route("/reset-deterministic", methods=["POST"])
def reset_deterministic():
    reset_latest_deterministic_submodel_json()
    return jsonify({"ok": True})


@app.route("/reset-submodels", methods=["POST"])
def reset_submodels():
    reset_latest_submodel_jsons()
    return jsonify({"ok": True})


@app.route("/update-aml", methods=["POST"])
def update_aml():
    payload = request.get_json(silent=True) or {}
    aml_model = payload.get("aml_model", "").strip()
    deterministic_context = payload.get("deterministic_context", "").strip()
    model_name = payload.get("model") or DEFAULT_MODEL

    if not aml_model:
        return jsonify({"error": "Please import an AML, XML, or SML file first."}), 400
    if not deterministic_context:
        return jsonify({"error": "Please generate the deterministic changes and TechnicalData first."}), 400

    try:
        available_models = get_ollama_models()
        if model_name not in available_models:
            model_name = DEFAULT_MODEL if DEFAULT_MODEL in available_models else available_models[0]

        updated_aml = update_aml_model(
            model_name,
            deterministic_context,
            aml_model,
        )
        return jsonify({
            "updated_aml_model": updated_aml,
            "source_name": payload.get("source_name", ""),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def is_aml_update_request(message: str) -> bool:
    """Return True when chat text asks to apply/update AML/XML from current UI context."""
    text = message or ""
    mentions_aml = re.search(r"\b(aml|xml|sml)\b", text, re.IGNORECASE)
    asks_update = re.search(r"\b(update|apply|modify|change|refresh)\b", text, re.IGNORECASE)
    mentions_change_data = re.search(r"\b(deterministic|technicaldata|technical data|submodel)\b", text, re.IGNORECASE)
    return bool(mentions_aml and asks_update and mentions_change_data)


@app.route("/chat", methods=["POST"])
def chat():
    global agent_short_memory_reset_done

    if not agent_short_memory_reset_done:
        reset_agent_short_memory()
        agent_short_memory_reset_done = True

    payload = request.get_json(silent=True) or {}
    user_message = payload["message"]
    model_name = payload.get("model") or DEFAULT_MODEL
    nondeterministic_context = payload.get("nondeterministic_context", "").strip()
    deterministic_context = payload.get("deterministic_context", "").strip()
    aml_model = payload.get("aml_model", "").strip()
    aml_source_name = payload.get("aml_source_name", "")
    use_direct_aml_route = is_aml_update_request(user_message)
    use_direct_deterministic_route = (
        not use_direct_aml_route
        and bool(nondeterministic_context)
        and bool(re.search(r"\bdeterministic\b", user_message, re.IGNORECASE))
    )

    try:
        available_models = get_ollama_models()
        if model_name not in available_models:
            model_name = DEFAULT_MODEL if DEFAULT_MODEL in available_models else available_models[0]

        if use_direct_aml_route:
            if not aml_model:
                return jsonify({
                    "reply": "<pre>Please import an AML, XML, or SML file first.</pre>",
                    "nondeterministic_submodel_json": "",
                    "deterministic_submodel_json": "",
                    "updated_aml_model": "",
                    "aml_source_name": aml_source_name,
                })
            if not deterministic_context:
                return jsonify({
                    "reply": "<pre>Please generate the Deterministic and TechnicalData submodels first.</pre>",
                    "nondeterministic_submodel_json": "",
                    "deterministic_submodel_json": "",
                    "updated_aml_model": "",
                    "aml_source_name": aml_source_name,
                })

            updated_aml = update_aml_model(
                model_name,
                deterministic_context,
                aml_model,
            )
            return jsonify({
                "reply": "<pre>Updated AML model applied to the UI AML box.</pre>",
                "nondeterministic_submodel_json": "",
                "deterministic_submodel_json": "",
                "updated_aml_model": updated_aml,
                "aml_source_name": aml_source_name,
            })

        if use_direct_deterministic_route:
            deterministic_json = convert_nondeterministic_to_deterministic(
                model_name,
                nondeterministic_context,
            )
            return jsonify({
                "reply": f"<pre>{html.escape(deterministic_json)}</pre>",
                "nondeterministic_submodel_json": "",
                "deterministic_submodel_json": deterministic_json,
            })

        main_agent = build_main_agent(model_name)
        question = HumanMessage(content=user_message)
        nondeterministic_calls_before = get_nondeterministic_agent_call_count()
        deterministic_calls_before = get_deterministic_agent_call_count()

        response = main_agent.invoke(
            {"messages": [question]},
            config={"configurable": {"thread_id": "main_agent"}},
        )

        messages = response["messages"]

        final_answer = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                final_answer = msg.content
                break

        # Convert Markdown → HTML
        html_answer = markdown.markdown(final_answer)
        nondeterministic_called = (
            get_nondeterministic_agent_call_count() > nondeterministic_calls_before
        )
        deterministic_called = (
            get_deterministic_agent_call_count() > deterministic_calls_before
        )

        return jsonify({
            "reply": html_answer,
            "nondeterministic_submodel_json": (
                get_last_nondeterministic_submodel_json()
                if nondeterministic_called
                else ""
            ),
            "deterministic_submodel_json": (
                get_last_deterministic_submodel_json()
                if deterministic_called
                else ""
            ),
        })

    except Exception as e:
        return jsonify({
            "reply": f"<pre>Error: {str(e)}</pre>",
            "nondeterministic_submodel_json": "",
            "deterministic_submodel_json": "",
        })
# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    host = "0.0.0.0"
    port = find_free_port(8094, host)
    print(f"LLM4ECM UI running locally at http://127.0.0.1:{port}", flush=True)
    print(f"LLM4ECM UI listening on all interfaces at http://0.0.0.0:{port}", flush=True)
    app.run(host=host, port=port, debug=False, use_reloader=False)