import rich


class LogColor:
    def warn(log: str):
        rich.print(f"[yellow]{log}[/yellow]")

    def info(log: str):
        rich.print(f"[blue]{log}[/blue]")

    def error(log: str):
        rich.print(f"[red]{log}[/red]")

    def regular(log: str):
        rich.print(log)
