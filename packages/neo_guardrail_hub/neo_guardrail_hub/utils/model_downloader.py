"""
Model Downloader for LLM Guard

Downloads HuggingFace models for LLM Guard scanners to local storage,
enabling faster initialization and offline operation.

Usage:
    downloader = LLMGuardModelDownloader(config_path="./configs", models_dir="./models/llm_guard")
    downloader.download_all_enabled()
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import yaml

from neo_guardrail_hub.providers.llm_guard_model_registry import (
    LLM_GUARD_MODELS,
    ModelInfo,
    get_model_info,
)


class LLMGuardModelDownloader:
    """Downloads LLM Guard models based on enabled scanners in config."""
    
    def __init__(self, config_path: Optional[str] = None, models_dir: str = "./models/llm_guard"):
        """Initialize the model downloader.
        
        Args:
            config_path: Path to config directory containing default.yaml (optional)
            models_dir: Directory where models will be downloaded
        """
        self.config_path = Path(config_path) if config_path else None
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_enabled_scanners(self) -> List[str]:
        """Parse config YAML to find all enabled LLM Guard scanners.
        
        Returns:
            List of scanner type strings (e.g., ["prompt_injection", "toxicity_input"])
        """
        if not self.config_path:
            print("⚠️  No config path provided, cannot parse enabled scanners")
            return []
            
        config_file = self.config_path / "default.yaml"
        
        if not config_file.exists():
            print(f"⚠️  Config file not found: {config_file}")
            return []
        
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  Failed to parse config: {e}")
            return []
        
        enabled_scanners = []
        
        # Check input scanners
        input_checks = config.get("guardrails", {}).get("input", {}).get("checks", [])
        for check in input_checks:
            if check.get("enabled") and check.get("provider") == "llm_guard":
                enabled_scanners.append(check["type"])
        
        # Check output scanners
        output_checks = config.get("guardrails", {}).get("output", {}).get("checks", [])
        for check in output_checks:
            if check.get("enabled") and check.get("provider") == "llm_guard":
                enabled_scanners.append(check["type"])
        
        return enabled_scanners
    
    def check_git_lfs(self) -> bool:
        """Check if git-lfs is installed.
        
        Returns:
            True if git-lfs is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["git", "lfs", "version"],
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def install_git_lfs(self) -> bool:
        """Attempt to install git-lfs.
        
        Returns:
            True if installation successful, False otherwise
        """
        print("\n📦 Installing git-lfs...")
        try:
            subprocess.run(["git", "lfs", "install"], check=True, capture_output=True)
            print("✓ git-lfs installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install git-lfs: {e}")
            print("Please install git-lfs manually:")
            print("  macOS: brew install git-lfs")
            print("  Ubuntu: sudo apt-get install git-lfs")
            print("  Windows: Download from https://git-lfs.github.com/")
            return False
    
    def download_model(self, model_info: ModelInfo) -> bool:
        """Download a single model using git clone.
        
        Args:
            model_info: ModelInfo object with download details
            
        Returns:
            True if download successful, False otherwise
        """
        if not model_info.requires_model:
            print(f"⏭️  {model_info.scanner_type}: No model required (rule-based)")
            return True
        
        target_dir = self.models_dir / model_info.local_dir_name
        
        # Skip if already exists
        if target_dir.exists() and (target_dir / "config.json").exists():
            print(f"✓ {model_info.scanner_type}: Already downloaded at {target_dir}")
            return True
        
        print(f"\n📥 Downloading {model_info.scanner_type}")
        print(f"   Repository: {model_info.hf_repo}")
        print(f"   Target: {target_dir}")
        
        try:
            # Clone the model repository from HuggingFace
            hf_url = f"https://huggingface.co/{model_info.hf_repo}"
            
            subprocess.run(
                ["git", "clone", hf_url, str(target_dir)],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Pull LFS files (models are stored as LFS objects)
            print(f"   Pulling LFS files...")
            subprocess.run(
                ["git", "lfs", "pull"],
                cwd=str(target_dir),
                check=True,
                capture_output=True,
                text=True
            )
            
            # Verify the download
            if not (target_dir / "config.json").exists():
                print(f"⚠️  Download completed but config.json not found")
                return False
            
            # Check if model file is actually downloaded (not just LFS pointer)
            model_file = target_dir / "model.safetensors"
            if model_file.exists() and model_file.stat().st_size < 1000:
                print(f"⚠️  model.safetensors looks like LFS pointer file")
                return False
            
            # Get directory size
            size_mb = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file()) / (1024 * 1024)
            
            print(f"✓ Successfully downloaded {model_info.scanner_type} ({size_mb:.1f} MB)")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download {model_info.scanner_type}")
            print(f"   Error: {e.stderr if hasattr(e, 'stderr') else str(e)}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error downloading {model_info.scanner_type}: {e}")
            return False
    
    def download_all_enabled(self) -> dict:
        """Download all models for enabled scanners.
        
        Returns:
            Dictionary with download statistics
        """
        # Check git-lfs
        if not self.check_git_lfs():
            print("⚠️  git-lfs not found")
            if not self.install_git_lfs():
                print("\n❌ Cannot proceed without git-lfs")
                return {"success": 0, "failed": 0, "skipped": 0}
        
        # Get enabled scanners
        enabled = self.parse_enabled_scanners()
        
        if not enabled:
            print("⚠️  No enabled LLM Guard scanners found in config")
            return {"success": 0, "failed": 0, "skipped": 0}
        
        print(f"\n{'='*70}")
        print(f"📋 LLM Guard Model Downloader")
        print(f"{'='*70}")
        print(f"Config: {self.config_path / 'default.yaml'}")
        print(f"Models directory: {self.models_dir.absolute()}")
        print(f"Found {len(enabled)} enabled LLM Guard scanners\n")
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for scanner_type in enabled:
            model_info = get_model_info(scanner_type)
            
            if not model_info:
                print(f"⚠️  {scanner_type}: Model info not found in registry")
                skipped_count += 1
                continue
            
            if self.download_model(model_info):
                success_count += 1
            else:
                failed_count += 1
        
        # Summary
        print(f"\n{'='*70}")
        print(f"📊 Download Summary")
        print(f"{'='*70}")
        print(f"✓ Successful: {success_count}")
        print(f"✗ Failed: {failed_count}")
        print(f"⏭️  Skipped: {skipped_count}")
        print(f"Total: {len(enabled)}\n")
        
        if success_count > 0:
            print(f"Models are ready to use! Set environment variable:")
            print(f"  export NEO_LLM_GUARD_MODELS_DIR={self.models_dir.absolute()}")
        
        return {
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count
        }
    
    def download_specific_models(self, scanner_types: List[str]) -> dict:
        """Download specific models by scanner type.
        
        Args:
            scanner_types: List of scanner types to download
            
        Returns:
            Dictionary with download statistics:
            - success: number of successful downloads
            - failed: number of failed downloads
            - skipped: number of skipped models (already downloaded)
            - total_size_mb: total size of downloaded models in MB
        """
        print(f"\n{'='*70}")
        print(f"📋 LLM Guard Model Downloader")
        print(f"{'='*70}")
        print(f"Models directory: {self.models_dir.absolute()}")
        print(f"Downloading {len(scanner_types)} specific models\n")
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        total_size_mb = 0.0
        
        for scanner_type in scanner_types:
            model_info = get_model_info(scanner_type)
            
            if not model_info:
                print(f"⚠️  {scanner_type}: Model info not found in registry")
                skipped_count += 1
                continue
            
            # Check if already downloaded
            target_dir = self.models_dir / model_info.local_dir_name
            if target_dir.exists():
                # Check if model file exists and is valid
                model_file = target_dir / "model.safetensors"
                config_file = target_dir / "config.json"
                
                if config_file.exists() and model_file.exists() and model_file.stat().st_size > 1000:
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    total_size_mb += size_mb
                    print(f"⏭️  {scanner_type}: Already downloaded ({size_mb:.1f} MB)")
                    skipped_count += 1
                    continue
            
            if self.download_model(model_info):
                success_count += 1
                # Calculate size
                model_file = target_dir / "model.safetensors"
                if model_file.exists():
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    total_size_mb += size_mb
            else:
                failed_count += 1
        
        print(f"\n{'='*70}")
        print(f"✓ Downloaded: {success_count}, Skipped: {skipped_count}, Failed: {failed_count}")
        print(f"  Total size: {total_size_mb:.1f} MB\n")
        
        return {
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total_size_mb": total_size_mb
        }
