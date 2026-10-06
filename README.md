# RestoreLab

**Motion deblurring e denoising: confronto tra metodi variazionali, Plug-and-Play e deep learning.**

RestoreLab è un progetto sperimentale di computational imaging per ricostruire immagini RGB degradate da sfocatura da movimento e rumore gaussiano. Attraverso notebook Jupyter, confronta regolarizzazione Total Variation, prior neurali e generativi e reti addestrate per il restauro, usando una degradazione condivisa e metriche di qualità dell'immagine.

## Il problema

L'obiettivo è stimare un'immagine pulita `x` a partire da un'osservazione degradata:

```text
y = Kx + rumore
```

`K` rappresenta il motion blur. Nell'implementazione le osservazioni vengono anche limitate all'intervallo `[0, 1]`.

| Parametro | Configurazione |
| --- | --- |
| Immagini | RGB, 256 × 256 pixel |
| Sfocatura | Motion blur, kernel 9 × 9, angolo 45° |
| Livelli di rumore | 0.005, 0.01, 0.05, 0.1 |
| Metriche | PSNR e SSIM; errore relativo (RE) in alcuni moduli |

La degradazione è definita in [utilities/degradation.py](utilities/degradation.py).

## Metodi implementati

| Metodo | Descrizione | Notebook |
| --- | --- | --- |
| Total Variation | Regolarizzazione variazionale con solver Chambolle–Pock | [tv_regularisation.ipynb](tv_regularisation.ipynb) |
| Plug-and-Play HQS | Half-Quadratic Splitting con denoiser DRUNet preaddestrato e congelato | [hqs_pnp_heuristic.ipynb](hqs_pnp_heuristic.ipynb) |
| HQSNet | Algoritmo HQS unrolled con parametri appresi e prior DRUNet congelato | [hqs_pnp_net.ipynb](hqs_pnp_net.ipynb) |
| NAFNet | Rete neurale addestrata per deblurring e denoising | [naf_net.ipynb](naf_net.ipynb) |
| Deep Generative Prior | Ottimizzazione del latente di BigGAN, con classe stimata tramite ResNet50 | [dgp_regularisation.ipynb](dgp_regularisation.ipynb) |
| GAN | Esperimento di addestramento di una rete generativa avversaria | [gan.ipynb](gan.ipynb) |

Il notebook [final_comparison.ipynb](final_comparison.ipynb) raccoglie confronti visivi e metriche per TV, PnP-HQS, NAFNet e DGP.

## Dataset

Gli esperimenti caricano tramite Hugging Face Datasets il dataset `benjamin-paine/imagenet-1k-256x256`. Le immagini vengono convertite in RGB, ridimensionate a 256 × 256 e trasformate in tensori.

Le dimensioni dei sottoinsiemi sono definite in [utilities/config.py](utilities/config.py):

- **Training:** 10.000 immagini.
- **Validazione:** 500 immagini.
- **Test:** 200 immagini.

Lo stesso file contiene `TO_RECONSTRUCT_INDEXES`, gli indici utilizzati per i confronti qualitativi. Il primo caricamento richiede accesso al dataset e spazio per la cache locale.

## Struttura del progetto

```text
Project/
├── IPPy/                      # Operatori, solver e metriche per problemi inversi
├── models/
│   ├── heuristic_tv_regularizer.py
│   ├── heuristic_hqs_pnp.py
│   ├── heuristic_dgp.py
│   ├── network_hqs_net.py
│   ├── network_unet.py         # Architettura usata per DRUNet
│   ├── network_gan.py
│   └── basicblock.py
├── utilities/
│   ├── config.py              # Dimensioni dei dataset e indici di test
│   ├── degradation.py         # Blur e rumore
│   ├── image_dataset.py       # Preprocessing delle immagini
│   └── plotter.py             # Visualizzazione e salvataggio dei confronti
├── weights/                   # Pesi preaddestrati e checkpoint
├── results/                   # Risultati degli esperimenti
├── SPECIFICA.pdf
├── tv_regularisation.ipynb
├── hqs_pnp_heuristic.ipynb
├── hqs_pnp_net.ipynb
├── naf_net.ipynb
├── dgp_regularisation.ipynb
├── gan.ipynb
└── final_comparison.ipynb
```

## Preparazione dell'ambiente

Il progetto utilizza Python e PyTorch. Una GPU compatibile con CUDA è consigliata per l'addestramento e per le ricostruzioni con prior generativi.

Esempio di ambiente di partenza in **PowerShell**, eseguendo i comandi dalla cartella `Project/`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision numpy matplotlib pillow scikit-image datasets tqdm jupyterlab ipykernel
.\.venv\Scripts\python.exe -m ipykernel install --user --name restorelab --display-name "Python (RestoreLab)"
```

Dipendenze aggiuntive per DGP e NAFNet, importate anche dal confronto finale:

```powershell
.\.venv\Scripts\python.exe -m pip install pytorch-pretrained-biggan focal-frequency-loss
```

Questi comandi elencano le dipendenze necessarie agli import principali; la cartella non contiene un ambiente con versioni bloccate e verificato per tutti i notebook. Per usare CUDA, scegli una build di PyTorch compatibile con l'hardware disponibile.

### Modulo NAFNet esterno alla cartella

`naf_net.ipynb` e `final_comparison.ipynb` importano `nafnet_heightmap.network_naf_net`. Nella struttura attuale questo modulo si trova nella directory superiore, accanto a `Project/`.

Per eseguire questi notebook mantenendo il kernel in `Project/`, aggiungi prima degli import:

```python
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
```

Se distribuisci soltanto la cartella `Project/`, devi includere anche il pacchetto `nafnet_heightmap` o adattare gli import: attualmente i due notebook dipendono da codice esterno a questa cartella.

## Esecuzione

Avvia JupyterLab dalla cartella `Project/`:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

1. Apri il notebook del metodo da eseguire e seleziona il kernel `Python (RestoreLab)`.
2. Mantieni `Project/` come directory di lavoro: gli import `models`, `utilities` e i percorsi `./weights/...` dipendono da questa scelta.
3. Controlla configurazione del dataset, dispositivo, parametri di training o ricostruzione e percorsi dei checkpoint.
4. Esegui le celle in ordine. Per i modelli appresi, esegui prima il training oppure prepara un checkpoint compatibile.
5. Esegui le celle di valutazione per visualizzare originale, osservazione degradata e ricostruzione con le rispettive metriche.

Per iniziare dal metodo variazionale, apri `tv_regularisation.ipynb`. Passa poi a PnP-HQS e ai modelli appresi; esegui il confronto finale dopo aver preparato le dipendenze e i pesi dei metodi coinvolti.

### Pesi e checkpoint

| Metodo | Risorsa attesa |
| --- | --- |
| PnP-HQS e HQSNet | `weights/DRUNet/drunet_color.pth` |
| HQSNet | Training configurato su `weights/HQSNet/HQS_checkpoint.pth`; valutazione sul corrispondente file `_best.pth` |
| NAFNet | `weights/NAFNet/NAFImgDeblur&Denoise.pth` |
| GAN | `weights/GAN/gan.pth` |
| DGP | BigGAN `biggan-deep-256` e ResNet50 con pesi ImageNet, caricati tramite le rispettive librerie |

La presenza delle directory non garantisce che tutti i pesi siano disponibili. Verifica i file richiesti prima della valutazione; il primo utilizzo dei modelli preaddestrati DGP può richiedere un download.

## Valutazione e riproducibilità

PSNR e SSIM confrontano la ricostruzione con l'immagine pulita. I notebook producono confronti visivi e, a seconda dell'esperimento, salvano immagini e checkpoint nei percorsi configurati.

Nei notebook TV, PnP-HQS e DGP il miglior parametro di regolarizzazione o schedule viene selezionato tramite PSNR rispetto alla ground truth. Questa valutazione presuppone quindi la disponibilità dell'immagine pulita; su dati reali senza riferimento serve un diverso criterio di selezione.

Per confronti riproducibili, registra configurazione della degradazione, seed, immagini selezionate, checkpoint e parametri di ricostruzione.

**Nota sul DGP:** l'introduzione del notebook menziona StyleGAN-XL, ma l'implementazione corrente in `models/heuristic_dgp.py` utilizza BigGAN. La descrizione di questo README segue il codice.
