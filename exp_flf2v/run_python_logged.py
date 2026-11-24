"""Merge Python stderr into stdout before running a script (PowerShell 5 safe)."""
import faulthandler
import runpy
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: run_python_logged.py script.py [arguments]')
    sys.stderr = sys.stdout
    faulthandler.enable(file=sys.stdout, all_threads=True)
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name='__main__')
