import sys
with open('err2.txt', 'w', encoding='utf-8') as f:
    try:
        from manhwa_app.app import MainWindow
        f.write('IMPORT_OK\n')
        print("IMPORT_OK")
    except Exception as e:
        import traceback
        traceback.print_exc(file=f)
        print("FAILED")
