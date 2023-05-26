# Stages
Stages werden in der dvc.yaml Datei erstellt. Die Stages verwenden die Python Skripte aus dem Ordner 'src', welche Modulare Funktionalitäten anbieten.
## DAG 
Um eine Übersicht über die Pipeline zu erhalten, kann der Befehl 'dvc dag' verwendet werden.

```mermaid
flowchart TD
        node1["aggregate_mean_median_std"]
        node2["correlations_pearson"]
        node3["dvclive"]
        node4["evaluate"]
        node5["fft"]
        node6["filter_bandpass_ord3_low01_high100"]
        node7["filter_highpass_ord3_cut01"]
        node8["filter_lowpass_ord3_cut100"]
        node9["moving_average_01s"]
        node10["pull_data_calibrated"]
        node11["resample_50Hz"]
        node12["segmentate_5s"]
        node13["stft"]
        node14["train_test_split_ratio02"]
        node15["transform_min_max"]
        node16["transform_pca"]
        node17["transform_standardize"]
        node3-->node4
        node10-->node11
        node11-->node12
        node12-->node1
        node12-->node2
        node12-->node5
        node12-->node6
        node12-->node7
        node12-->node8
        node12-->node9
        node12-->node13
        node12-->node14
        node12-->node15
        node12-->node16
        node12-->node17
        
        node1 -->|opt.| node3
        node2 -->|opt.| node3
        node5 -->|opt.| node3
        node6 -->|opt.| node3
        node7 -->|opt.| node3
        node8 -->|opt.| node3
        node9 -->|opt.| node3
        node13 -->|opt.| node3
        node15 -->|opt.| node3
        node16 -->|opt.| node3
        node17 -->|opt.| node3
        
        node14-->node3
        node14-->node15
        node14-->node16
        node14-->node17
```

## Skripts für Tabulare Daten
Hier werden alle Skripts aufgelistet, welche Tabulare Daten (.parquet) als Input verwenden.
| Skript                | Beschreibung                                         | Output Typ       |
| --------------------- | ---------------------------------------------------- | ---------------- |
| pull_data.py          | Holt die gesammelten Daten von der Datenbank         | Tabular          |
| resample.py           | Resamplet die Daten auf eine bestimmte Frequenz      | Tabular          |
| segmentate.py         | Segmentiert die Daten in bestimmte Zeitfenster       | Segmente         |

## Skripts für Segmentierte Daten
Hier werden alle Skripts aufgelistet, welche Segmentierte Daten (.dill) als Input verwenden.
| Skript                | Beschreibung                                        | Output Typ             |
| --------------------- | --------------------------------------------------- | ---------------------- |
| aggregate.py          | Aggregiert die Daten anhand einer Aggregationsliste | Segmente               |
| butterworth_filter.py | Filtert die Daten mit einem Butterworth Filter      | Segmente               |
| correlations.py       | Berechnet die Korrelationen zwischen den Features   | Tabular                |
| fft.py                | Führt eine FFT auf den Daten durch                  | Segmente               |
| moving_average.py     | Berechnet den Moving Average der Daten              | Segmente               |
| train_test_split.py   | Teilt die Daten in Trainings- und Testdaten auf     | JSON Element           |
| transform.py          | Verarbeitet die Daten mit Scaler oder PCA           | Segmente + Transformer |

## Skripts für Trainieren und Evaluieren von Modellen
Hier werden alle Skripts aufgelistet, welche für das Trainieren und Evaluieren von Modellen verwendet werden.
| Skript          | Beschreibung                                       | Output Typ               |
| --------------- | -------------------------------------------------- | ------------------------ |
| evaluate.py     | Erstellt anhand der Metriken alle benötigten Plots | Plots                    |
| model_runner.py | Führt ein Modell aus                               | Metriken und Vorhersagen |
