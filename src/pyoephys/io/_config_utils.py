"""
Configuration and interactive prompt utilities.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
try:
    from tkinter import Tk, filedialog, simpledialog, messagebox
except Exception:
    Tk = filedialog = simpledialog = messagebox = None


def load_simple_config(config_path: Path | str) -> Dict[str, Any]:
    config = {}
    config_path = Path(config_path)
    
    if not config_path.exists():
        return config
    
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                config[key] = value
    return config


def save_simple_config(config: Dict[str, Any], config_path: Path | str, header: str = "Configuration File"):
    config_path = Path(config_path)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(f"# {header}\n")
        f.write("# Automatically generated - edit as needed\n\n")
        for key, value in config.items():
            if isinstance(value, bool):
                value = str(value).lower()
            f.write(f"{key}={value}\n")


def prompt_directory(title: str = "Select Directory", initial_dir: Optional[str] = None, use_terminal: bool = False) -> Optional[str]:
    if use_terminal:
        default = initial_dir or os.getcwd()
        result = input(f"{title} [{default}]: ").strip()
        path = result if result else default
        return path if os.path.isdir(path) else None
    
    try:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        directory = filedialog.askdirectory(title=title, initialdir=initial_dir or os.getcwd())
        root.destroy()
        return directory if directory else None
    except Exception:
        return prompt_directory(title, initial_dir, use_terminal=True)


def prompt_file(title: str = "Select File", initial_dir: Optional[str] = None, 
                filetypes: Optional[list] = None) -> Optional[str]:
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    filetypes = filetypes or [("All files", "*.*")]
    file_path = filedialog.askopenfilename(title=title, initialdir=initial_dir or os.getcwd(), filetypes=filetypes)
    root.destroy()
    return file_path if file_path else None


def prompt_text(title: str, prompt: str, initial_value: str = "", use_terminal: bool = False) -> Optional[str]:
    if use_terminal:
        if initial_value:
            result = input(f"{prompt} [{initial_value}]: ").strip()
            return result if result else initial_value
        else:
            result = input(f"{prompt}: ").strip()
            return result if result else None
    
    try:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        result = simpledialog.askstring(title=title, prompt=prompt, initialvalue=initial_value)
        root.destroy()
        return result
    except Exception:
        return prompt_text(title, prompt, initial_value, use_terminal=True)


def prompt_yes_no(title: str, message: str) -> bool:
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    result = messagebox.askyesno(title=title, message=message)
    root.destroy()
    return result


def get_or_prompt_value(
    arg_value: Any,
    config: Dict[str, Any],
    key: str,
    prompt_func,
    required: bool = True,
    **prompt_kwargs
) -> Tuple[Any, bool]:
    if arg_value:
        return arg_value, False
    if key in config:
        return config[key], False
    value = prompt_func(**prompt_kwargs)
    if not value and required:
        raise ValueError(f"{key} is required but not provided")
    return value, True
