def minutes_sec_formating(milli_second: int | float) -> str:
    sec = milli_second // 1000
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"
