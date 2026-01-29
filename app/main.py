import sys
import traceback

def main() -> None:
    use_tk = False
    
    # Check if user explicitly wants Tkinter (optional arg)
    if "--tk" in sys.argv:
        use_tk = True

    if not use_tk:
        try:
            from PySide6.QtWidgets import QApplication
            from .ui.main_window import MainWindow
            
            app = QApplication(sys.argv)
            app.setApplicationName("配准可视化工具")
            win = MainWindow()
            win.show()
            sys.exit(app.exec())
        except ImportError:
            print("Warning: PySide6 not found or DLL load failed. Falling back to Tkinter...")
            use_tk = True
        except Exception:
            print("Warning: Error initializing PySide6. Falling back to Tkinter...")
            traceback.print_exc()
            use_tk = True

    if use_tk:
        _run_tk()

def _run_tk():
    try:
        from .ui_tk.main_window import MainWindow as TkMainWindow
        print("Starting Tkinter interface...")
        app = TkMainWindow()
        app.mainloop()
    except Exception:
        print("Error: Failed to load Tkinter interface.")
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)
