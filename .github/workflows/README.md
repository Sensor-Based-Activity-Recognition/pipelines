# Workflows
Hier werden die Workflows für die GitHub Actions definiert.

## Konfigurierte Runner
Hier werden die konfigurierten Runner aufgelistet.
### amd-ryzen9-7900x-nvidia-rtx-4070ti-gabo
Dieser Runner wird von Gabriel betrieben und läuft in einer Docker-Umgebung. Die Umgebung hat Zugriff auf eine Ryzen 9 7900X CPU mit 12 Kernen und 24 Threads, 32 GB DDR5-6000 RAM und eine Nvidia RTX 4070 Ti Grafikkarte. Der Runner wird mit dem Docker-Image "iterativeai/cml:0-dvc2-base1-gpu" erstellt. Die Grafikkarte verwendet Treiberversion 530.30.02 und die CUDA Version 12.1.

## Konfigurierte Workflows
Hier werden die konfigurierten Workflows aufgelistet.
### run_pipelines.yaml
Dieser Workflow wird automatisch nach jedem Commit durchgeführt. Seine Aufgabe besteht darin, eine Umgebung für die Ausführung der Pipelines zu erstellen, die Pipelines auszuführen und bei Änderungen in den Modellmetriken oder Parametern einen Bericht zu erstellen. Zudem werden die Metriken anschließend auf DVC hochgeladen und ein Pull Request erstellt, falls die Änderungen angenommen werden sollen.
Dieser Workflow greift auf den Runner "amd-ryzen9-7900x-nvidia-rtx-4070ti-gabo" zu.