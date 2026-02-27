
import gradio as gr
from pathlib import Path
import sys
import json
import os
import tempfile
import atexit
import shutil
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.orchestrator import RehabCoachAgent

TEMP_DIR = tempfile.mkdtemp(prefix="rehabai_")

def cleanup_on_exit():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f"Cleaned up temp directory: {TEMP_DIR}")

atexit.register(cleanup_on_exit)

print("Initializing RehabAI agent...")
agent = RehabCoachAgent(max_steps=3)
print("Agent ready\n")


def format_issues(issues):
    """Format detected issues with color-coded severity."""
    if not issues:
        return """
        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white; text-align: center;">
            <h2>✅ Excellent Form!</h2>
            <p style="font-size: 16px;">No major biomechanical issues detected. Keep up the good work!</p>
        </div>
        """
    
    html = '<div style="padding: 15px;">'
    
    for issue in issues:
        severity = issue.get('severity', 'mild')
        severity_colors = {
            'severe': {'bg': '#fee2e2', 'border': '#dc2626', 'icon': '🔴', 'text': '#7f1d1d'},
            'moderate': {'bg': '#fef3c7', 'border': '#f59e0b', 'icon': '🟡', 'text': '#78350f'},
            'mild': {'bg': '#d1fae5', 'border': '#10b981', 'icon': '🟢', 'text': '#065f46'}
        }
        
        colors = severity_colors.get(severity, severity_colors['mild'])
        issue_title = issue['type'].replace('_', ' ').title()
        
        html += f"""
        <div style="background: {colors['bg']}; border-left: 4px solid {colors['border']}; 
                    padding: 15px; margin-bottom: 15px; border-radius: 8px;">
            <h3 style="color: {colors['text']}; margin: 0 0 10px 0;">
                {colors['icon']} {issue_title} - {severity.upper()}
            </h3>
            <p style="color: {colors['text']}; margin: 0; line-height: 1.6;">
                {issue['description']}
            </p>
        </div>
        """
    
    html += '</div>'
    return html

def convert_to_serializable(obj):
    """Convert numpy/complex types to JSON-serializable types."""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def format_metrics(biomech_data):
    """Format metrics with visual score indicator."""
    quality = biomech_data.get('quality_score', 0)
    reps = biomech_data.get('rep_count', 0)
    duration = biomech_data.get('duration_sec', 0)
    
    if quality >= 8:
        score_emoji = '🎯'
    elif quality >= 6:
        score_emoji = '⚠️'
    else:
        score_emoji = '🔴'
    
    html = f"""
    <div style="padding: 20px;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; color: white; margin-bottom: 20px;">
            <h1 style="margin: 0 0 10px 0; font-size: 48px;">{score_emoji} {quality}/10</h1>
            <p style="margin: 0; font-size: 18px; opacity: 0.9;">Overall Movement Quality</p>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
            <div style="background: #f3f4f6; padding: 20px; border-radius: 10px; text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: #1f2937;">{reps}</div>
                <div style="color: #6b7280; margin-top: 5px;">Reps Detected</div>
            </div>
            <div style="background: #f3f4f6; padding: 20px; border-radius: 10px; text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: #1f2937;">{duration:.1f}s</div>
                <div style="color: #6b7280; margin-top: 5px;">Duration</div>
            </div>
        </div>
    """
    
    metrics = biomech_data.get('metrics', {})
    if metrics:
        html += """
        <div style="background: white; padding: 20px; border-radius: 10px; border: 1px solid #e5e7eb;">
            <h3 style="margin: 0 0 15px 0; color: #1f2937 !important;">Range of Motion Analysis</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: #f9fafb;">
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e5e7eb; color: #1f2937 !important;">Measurement</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #e5e7eb; color: #1f2937 !important;">Left</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #e5e7eb; color: #1f2937 !important;">Right</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #e5e7eb; color: #1f2937 !important;">Difference</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict):
                left = metric_value.get('left', 'N/A')
                right = metric_value.get('right', 'N/A')
                metric_display = metric_name.replace('_', ' ').title()
                
                # Convert strings to float if needed
                try:
                    left_num = float(left) if isinstance(left, str) else left
                    right_num = float(right) if isinstance(right, str) else right
                    
                    diff = abs(left_num - right_num)
                    diff_color = '#dc2626' if diff > 15 else '#f59e0b' if diff > 10 else '#10b981'
                    
                    html += f"""
                    <tr>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #1f2937 !important;">{metric_display}</td>
                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #e5e7eb; color: #1f2937 !important;">{left_num:.1f}°</td>
                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #e5e7eb; color: #1f2937 !important;">{right_num:.1f}°</td>
                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #e5e7eb; color: {diff_color} !important; font-weight: bold;">{diff:.1f}°</td>
                    </tr>
                    """
                except (ValueError, TypeError):
                    # If conversion fails, show as-is
                    html += f"""
                    <tr>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #1f2937 !important;">{metric_display}</td>
                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #e5e7eb; color: #1f2937 !important;">{left}</td>
                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #e5e7eb; color: #1f2937 !important;">{right}</td>
                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #e5e7eb; color: #1f2937 !important;">-</td>
                    </tr>
                    """
        
        html += """
                </tbody>
            </table>
        </div>
        """
    
    html += '</div>'
    return html


def format_evidence(evidence_chunks):
    """Format evidence with relevance scores."""
    if not evidence_chunks:
        return "<p style='color: #6b7280; text-align: center; padding: 20px;'>No research evidence retrieved</p>"
    
    html = f"""
    <div style="padding: 15px;">
        <div style="background: #eff6ff; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #3b82f6;">
            <p style="margin: 0; color: #1e40af !important; font-weight: 600;">
                📚 <strong style="color: #1e40af !important;">{len(evidence_chunks)} research sources</strong> used to support these recommendations
            </p>
        </div>
    """
    
    for i, chunk in enumerate(evidence_chunks[:3], 1):
        score = chunk.get('relevance_score', 0)
        text = chunk.get('text', '')[:300]
        
        score_color = '#10b981' if score > 0.6 else '#f59e0b' if score > 0.4 else '#6b7280'
        
        html += f"""
        <div style="background: white; border: 1px solid #e5e7eb; padding: 20px; 
                    margin-bottom: 15px; border-radius: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <strong style="color: #1f2937 !important;">Source {i}</strong>
                <span style="background: {score_color}; color: white; padding: 4px 12px; 
                            border-radius: 12px; font-size: 12px; font-weight: bold;">
                    Relevance: {score:.2f}
                </span>
            </div>
            <p style="color: #374151 !important; line-height: 1.6; margin: 0;">
                {text}...
            </p>
        </div>
        """
    
    html += '</div>'
    return html


def export_results(biomech_data, coaching_plan, issues):
    """Export results as JSON with proper serialization."""
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "biomechanics": {
            "quality_score": convert_to_serializable(biomech_data.get('quality_score')),
            "rep_count": convert_to_serializable(biomech_data.get('rep_count')),
            "duration_sec": convert_to_serializable(biomech_data.get('duration_sec')),
            "issues": convert_to_serializable(issues),
            "metrics": convert_to_serializable(biomech_data.get('metrics', {}))
        },
        "coaching_plan": coaching_plan
    }
    
    filename = f"rehabai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(TEMP_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return filepath


def analyze_video_enhanced(video_file, progress=gr.Progress()):
    """Enhanced analysis with progress tracking."""
    if video_file is None:
        return (
            "<p style='text-align: center; color: #6b7280; padding: 40px;'>Upload a video to begin analysis</p>",
            "", "", "", 
            "<p style='text-align: center; color: #ef4444;'>Please upload a video file</p>",
            None
        )
    
    try:
        progress(0.0, desc="Starting analysis...")
        print(f"Processing video: {video_file}")
        
        progress(0.2, desc="Running biomechanics analysis...")
        result = agent.run(video_file)
        
        progress(0.8, desc="Generating coaching plan...")
        
        biomech = result.get('biomechanics', {})
        issues = biomech.get('issues', [])
        coaching = result.get('coaching_plan', '')
        evidence = result.get('evidence', [])
        
        progress(0.95, desc="Finalizing results...")
        
        issues_html = format_issues(issues)
        metrics_html = format_metrics(biomech)
        evidence_html = format_evidence(evidence)
        
        export_file = export_results(biomech, coaching, issues)
        
        progress(1.0, desc="Complete!")
        
        status = f"""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h3 style="margin: 0 0 10px 0;">Analysis Complete!</h3>
            <p style="margin: 0; opacity: 0.9;">
                Detected {len(issues)} issue(s) | Used {len(evidence)} research sources
            </p>
        </div>
        """
        
        return issues_html, metrics_html, coaching, evidence_html, status, export_file
        
    except Exception as e:
        error_html = f"""
        <div style="background: #fee2e2; border: 1px solid #dc2626; padding: 20px; 
                    border-radius: 10px; color: #7f1d1d;">
            <h3 style="margin: 0 0 10px 0;">Error During Analysis</h3>
            <p style="margin: 0; font-family: monospace; font-size: 14px;">{str(e)}</p>
        </div>
        """
        return "", "", "", "", error_html, None


custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="RehabAI") as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 30px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; margin-bottom: 30px;">
        <h1 style="color: white; font-size: 48px; margin: 0 0 10px 0;">🏋️ RehabAI</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 20px; margin: 0;">
            AI-Powered Movement Analysis Coach
        </p>
    </div>
    
    <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 20px; 
                border-radius: 10px; margin-bottom: 30px;">
        <h3 style="margin: 0 0 10px 0; color: #78350f;">⚠️ Important Medical Disclaimer</h3>
        <p style="margin: 0; color: #78350f; line-height: 1.6;">
            This tool is for <strong style="color: #78350f;">educational purposes only</strong> and should NOT replace professional medical advice.
            If you have pain, injury, or medical conditions, consult a qualified healthcare provider before training.
        </p>
    </div>
    """)
    
    gr.Markdown("""
    ### 📹 Camera Setup Guidelines
    
    For best results:
    - **Position:** Front view (face the camera)
    - **Distance:** 6-10 feet from camera
    - **Framing:** Full body visible (head to feet)
    - **Lighting:** Good, even lighting
    - **Reps:** Perform 3-5 slow, controlled reps
    """)
    
    video_input = gr.Video(
        label="Upload Your Squat Video",
        sources=["upload", "webcam"],
        height=400
    )
    
    with gr.Row():
        analyze_btn = gr.Button(
            "🔍 Analyze Movement",
            variant="primary",
            size="lg",
            scale=3
        )
        clear_btn = gr.ClearButton(
            [video_input],
            value="Clear",
            size="lg",
            scale=1
        )
    
    status_output = gr.HTML(
        "<p style='text-align: center; color: #6b7280; padding: 40px;'>Upload a video and click 'Analyze Movement' to begin</p>"
    )
    
    export_output = gr.File(label="Download Results (JSON)", visible=False)
    
    gr.Markdown("---")
    
    with gr.Tabs():
        with gr.Tab("🎯 Issues & Metrics"):
            with gr.Row():
                with gr.Column(scale=1):
                    issues_output = gr.HTML()
                
                with gr.Column(scale=1):
                    metrics_output = gr.HTML()
        
        with gr.Tab("📝 Coaching Plan"):
            coaching_output = gr.Markdown()
        
        with gr.Tab("📚 Research Evidence"):
            evidence_output = gr.HTML()
    
    analyze_btn.click(
        fn=analyze_video_enhanced,
        inputs=[video_input],
        outputs=[
            issues_output,
            metrics_output,
            coaching_output,
            evidence_output,
            status_output,
            export_output
        ]
    )

    gr.Markdown("---")
    gr.Markdown("### About RehabAI")

    gr.HTML("""
    <div style="background: #f9fafb; padding: 30px; border-radius: 15px;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;">
            <div style="background: white; padding: 20px; border-radius: 10px; border: 1px solid #e5e7eb;">
                <h4 style="color: #667eea !important; margin: 0 0 10px 0;">🎥 Computer Vision</h4>
                <p style="color: #1f2937 !important; margin: 0; line-height: 1.6; font-weight: 500;">
                    MediaPipe Pose for real-time biomechanics analysis
                </p>
            </div>
            <div style="background: white; padding: 20px; border-radius: 10px; border: 1px solid #e5e7eb;">
                <h4 style="color: #667eea !important; margin: 0 0 10px 0;">📚 RAG System</h4>
                <p style="color: #1f2937 !important; margin: 0; line-height: 1.6; font-weight: 500;">
                    Retrieval-Augmented Generation for evidence-based recommendations
                </p>
            </div>
            <div style="background: white; padding: 20px; border-radius: 10px; border: 1px solid #e5e7eb;">
                <h4 style="color: #667eea !important; margin: 0 0 10px 0;">🤖 AI Coaching</h4>
                <p style="color: #1f2937 !important; margin: 0; line-height: 1.6; font-weight: 500;">
                    Gemini 2.5 Flash for personalized coaching generation
                </p>
            </div>
        </div>
        
        <div style="background: white; padding: 20px; border-radius: 10px; border: 1px solid #e5e7eb;">
            <h4 style="margin: 0 0 15px 0; color: #1f2937 !important;">🔒 Privacy & Security</h4>
            <div style="line-height: 2.0;">
                <div style="color: #1f2937 !important; font-weight: 500; margin-bottom: 8px;">
                    ✓ Videos processed locally on your device
                </div>
                <div style="color: #1f2937 !important; font-weight: 500; margin-bottom: 8px;">
                    ✓ Temporary files deleted immediately after analysis
                </div>
                <div style="color: #1f2937 !important; font-weight: 500; margin-bottom: 8px;">
                    ✓ No video data stored or uploaded to cloud
                </div>
                <div style="color: #1f2937 !important; font-weight: 500;">
                    ✓ Only anonymized biomechanics data sent to AI for coaching
                </div>
            </div>
        </div>
    </div>
    """)




if __name__ == "__main__":
    print("\nRehabAI - AI Movement Analysis Coach")
    print(f"Secure temp directory: {TEMP_DIR}")
    print("Starting Gradio interface...\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )