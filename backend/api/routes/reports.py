from fastapi import APIRouter, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backend.api.models.requests import ReportGenerationRequest
from backend.api.models.responses import ReportResponse

router = APIRouter()

# Create reports directory if it doesn't exist
REPORTS_DIR = Path("data/outputs/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/generate", response_model=ReportResponse)
async def generate_audit_report(
    background_tasks: BackgroundTasks,
    audit_type: str = Form(...),
    audit_results: str = Form(...),
    report_format: str = Form("pdf"),
    include_visualizations: bool = Form(True),
    include_recommendations: bool = Form(True)
):
    """
    Generate comprehensive audit report
    """
    
    try:
        # Parse audit results
        results_data = json.loads(audit_results)
        
        # Generate report ID
        report_id = f"{audit_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate report content
        report_content = _generate_report_content(
            audit_type, results_data, include_recommendations
        )
        
        # Save report based on format
        if report_format.lower() == "pdf":
            filename = f"{report_id}.pdf"
            filepath = REPORTS_DIR / filename
            _save_as_pdf(report_content, filepath)
        
        elif report_format.lower() == "html":
            filename = f"{report_id}.html"
            filepath = REPORTS_DIR / filename
            _save_as_html(report_content, filepath)
        
        else:  # Default to text
            filename = f"{report_id}.txt"
            filepath = REPORTS_DIR / filename
            _save_as_text(report_content, filepath)
        
        # Add to background tasks for cleanup (optional)
        background_tasks.add_task(_cleanup_old_reports)
        
        return ReportResponse(
            report_id=report_id,
            filename=filename,
            format=report_format,
            size=filepath.stat().st_size,
            generated_at=datetime.now().isoformat(),
            download_url=f"/api/v1/reports/download/{filename}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.get("/download/{filename}")
async def download_report(filename: str):
    """
    Download generated report
    """
    
    filepath = REPORTS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type='application/octet-stream'
    )

@router.get("/list")
async def list_reports():
    """
    List all available reports
    """
    
    reports = []
    
    for report_file in REPORTS_DIR.glob("*"):
        if report_file.is_file():
            stat = report_file.stat()
            reports.append({
                'filename': report_file.name,
                'size': stat.st_size,
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'format': report_file.suffix[1:] if report_file.suffix else 'unknown',
                'download_url': f"/api/v1/reports/download/{report_file.name}"
            })
    
    # Sort by creation time (newest first)
    reports.sort(key=lambda x: x['created_at'], reverse=True)
    
    return {"reports": reports, "total_count": len(reports)}

@router.delete("/delete/{filename}")
async def delete_report(filename: str):
    """
    Delete a specific report
    """
    
    filepath = REPORTS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    try:
        filepath.unlink()
        return {"message": f"Report {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete report: {str(e)}")

@router.post("/summary")
async def generate_summary_report(
    audit_results_list: str = Form(...),
    time_period: str = Form("30_days")
):
    """
    Generate summary report across multiple audits
    """
    
    try:
        # Parse multiple audit results
        all_results = json.loads(audit_results_list)
        
        # Generate summary statistics
        summary_stats = _calculate_summary_statistics(all_results, time_period)
        
        # Generate summary report
        summary_content = _generate_summary_report_content(summary_stats, time_period)
        
        # Save summary report
        report_id = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filename = f"{report_id}.txt"
        filepath = REPORTS_DIR / filename
        
        _save_as_text(summary_content, filepath)
        
        return {
            'report_id': report_id,
            'filename': filename,
            'summary_stats': summary_stats,
            'download_url': f"/api/v1/reports/download/{filename}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary report generation failed: {str(e)}")

def _generate_report_content(audit_type: str, results: Dict[str, Any], include_recommendations: bool) -> str:
    """Generate report content based on audit type and results"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    content = f"""
AI ETHICS TOOLKIT - {audit_type.upper()} AUDIT REPORT
{'=' * 60}

Generated: {timestamp}
Audit Type: {audit_type.title()}

EXECUTIVE SUMMARY
-----------------
"""
    
    # Add type-specific summary
    if audit_type == "bias":
        overall_score = results.get('overall_bias_score', 0)
        ethics_score = results.get('ethics_score', 0)
        content += f"""
Overall Bias Score: {overall_score:.3f}
Ethics Score: {ethics_score}/10
Bias Detected: {'Yes' if overall_score > 0.1 else 'No'}
"""
    
    elif audit_type == "privacy":
        privacy_score = results.get('overall_score', 0)
        issues_found = len(results.get('findings', []))
        content += f"""
Privacy Score: {privacy_score}/10
Issues Found: {issues_found}
Risk Level: {results.get('summary', {}).get('risk_level', 'Unknown')}
"""
    
    # Add detailed findings
    content += "\nDETAILED FINDINGS\n"
    content += "-" * 17 + "\n"
    
    findings = results.get('findings', []) or results.get('hallucinations', [])
    for i, finding in enumerate(findings, 1):
        content += f"\nFinding #{i}:\n"
        content += f"Type: {finding.get('type', 'Unknown')}\n"
        content += f"Severity: {finding.get('severity', 'Unknown')}\n"
        content += f"Description: {finding.get('description', 'No description')}\n"
    
    # Add recommendations if requested
    if include_recommendations:
        recommendations = results.get('recommendations', [])
        if recommendations:
            content += "\nRECOMMENDATIONS\n"
            content += "-" * 15 + "\n"
            for i, rec in enumerate(recommendations, 1):
                # Clean up formatting
                clean_rec = rec.replace('**', '').replace('*', '').strip()
                content += f"{i}. {clean_rec}\n"
    
    return content

def _save_as_text(content: str, filepath: Path):
    """Save content as text file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def _save_as_html(content: str, filepath: Path):
    """Save content as HTML file"""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI Ethics Audit Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2E5AAC; }}
        h2 {{ color: #1E3A8A; }}
        pre {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <pre>{content}</pre>
</body>
</html>
"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

def _save_as_pdf(content: str, filepath: Path):
    """Save content as PDF file (simplified - would use proper PDF library in production)"""
    # For now, save as text with .pdf extension
    # In production, use libraries like reportlab or weasyprint
    _save_as_text(content, filepath)

def _calculate_summary_statistics(all_results: List[Dict[str, Any]], time_period: str) -> Dict[str, Any]:
    """Calculate summary statistics across multiple audit results"""
    
    if not all_results:
        return {}
    
    total_audits = len(all_results)
    
    # Count by type
    audit_types = {}
    total_issues = 0
    ethics_scores = []
    
    for result in all_results:
        audit_type = result.get('audit_type', 'unknown')
        audit_types[audit_type] = audit_types.get(audit_type, 0) + 1
        
        # Count issues/findings
        issues = len(result.get('findings', [])) + len(result.get('hallucinations', []))
        total_issues += issues
        
        # Collect ethics scores
        if 'ethics_score' in result:
            ethics_scores.append(result['ethics_score'])
        elif 'overall_score' in result:
            ethics_scores.append(result['overall_score'])
    
    avg_ethics_score = sum(ethics_scores) / len(ethics_scores) if ethics_scores else 0
    
    return {
        'time_period': time_period,
        'total_audits': total_audits,
        'audit_types': audit_types,
        'total_issues_found': total_issues,
        'average_ethics_score': round(avg_ethics_score, 2),
        'summary_date': datetime.now().isoformat()
    }

def _generate_summary_report_content(stats: Dict[str, Any], time_period: str) -> str:
    """Generate summary report content"""
    
    content = f"""
AI ETHICS TOOLKIT - SUMMARY REPORT
===================================

Time Period: {time_period.replace('_', ' ').title()}
Generated: {stats.get('summary_date', 'Unknown')}

SUMMARY STATISTICS
------------------
Total Audits Conducted: {stats.get('total_audits', 0)}
Total Issues Found: {stats.get('total_issues_found', 0)}
Average Ethics Score: {stats.get('average_ethics_score', 0)}/10

AUDIT BREAKDOWN BY TYPE
-----------------------
"""
    
    audit_types = stats.get('audit_types', {})
    for audit_type, count in audit_types.items():
        content += f"{audit_type.title()}: {count} audits\n"
    
    return content

def _cleanup_old_reports():
    """Clean up old report files (background task)"""
    
    # Remove reports older than 30 days
    cutoff_time = datetime.now().timestamp() - (30 * 24 * 60 * 60)
    
    for report_file in REPORTS_DIR.glob("*"):
        if report_file.is_file() and report_file.stat().st_ctime < cutoff_time:
            try:
                report_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors
