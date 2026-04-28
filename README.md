# Zola AI — Giocatore Ibrido v2 (`playerPRR`)

**File:** `playerPRR.py`  
**Gioco:** Zola (scacchiera 8×8, due giocatori)  
**Algoritmo di base:** Minimax con Alpha-Beta Pruning + Iterative Deepening

---

## Indice

1. [Il gioco Zola — Contesto](#1-il-gioco-zola--contesto)
2. [Regole di gioco strategiche](#2-regole-di-gioco-strategiche)
3. [Architettura del giocatore](#3-architettura-del-giocatore)
4. [Sistema dei livelli di distanza](#4-sistema-dei-livelli-di-distanza)
5. [Funzione di valutazione euristica](#5-funzione-di-valutazione-euristica)
6. [Componenti euristiche nel dettaglio](#6-componenti-euristiche-nel-dettaglio)
7. [Ordinamento delle mosse](#7-ordinamento-delle-mosse)
8. [Algoritmo Alpha-Beta con gestione del timeout](#8-algoritmo-alpha-beta-con-gestione-del-timeout)
9. [Iterative Deepening](#9-iterative-deepening)
10. [Il fix principale rispetto alla v1](#10-il-fix-principale-rispetto-alla-v1)
11. [Costanti e pesi](#11-costanti-e-pesi)
12. [Entry point — `playerStrategy`](#12-entry-point--playerstrategy)
13. [Flusso completo di esecuzione](#13-flusso-completo-di-esecuzione)

---

## 1. Il gioco Zola — Contesto

Zola è un gioco a due giocatori (Red e Blue) su una scacchiera 8×8. La scacchiera parte **piena**, con le pedine disposte a scacchiera alternata.

Le mosse possibili sono di due tipi:

- **Mossa non catturante (movimento):** una pedina si sposta in una cella adiacente (8 direzioni) che sia **vuota** e si trovi a un livello di distanza **maggiore** rispetto alla cella di partenza (ovvero la pedina si allontana dal centro verso la periferia).
- **Mossa catturante:** una pedina si muove come una regina degli scacchi (in linea retta, attraversando celle vuote) fino alla prima pedina avversaria incontrata. La destinazione deve avere livello **minore o uguale** rispetto alla partenza (la pedina catturante rimane nella stessa "fascia" o si avvicina al centro).

Se un giocatore non ha mosse legali, **salta il turno** automaticamente. Vince chi cattura **tutte** le pedine avversarie.

---

## 2. Regole di gioco strategiche

Dalla struttura dell'euristica di `playerPRR` emergono principi strategici precisi che guidano ogni decisione. Queste non sono semplici linee guida generali, ma regole ricavate direttamente dai pesi e dalle funzioni di valutazione implementate.

### Regola 1 — Conserva le pedine sopra ogni altra considerazione

Il peso assegnato alla differenza di materiale (`_W_PIECES = 80`) supera di gran lunga tutti gli altri criteri combinati. Ne consegue una regola fondamentale:

> **Non sacrificare mai una pedina in cambio di un vantaggio posizionale, di mobilità o di pressione tattica.** Perdere una pedina in più dell'avversario porta una penalità di 80 punti, che nessuna combinazione degli altri fattori riesce a compensare.

---

### Regola 2 — Occupa gli angoli e la periferia

La valutazione posizionale usa il **quadrato del livello** di ogni cella occupata (`lv²`). Questo crea una gerarchia netta:

| Posizione         | Livello | Valore posizionale |
|-------------------|---------|--------------------|
| Centro            | 1       | 1                  |
| Zona intermedia   | 5       | 25                 |
| Bordo             | 8       | 64                 |
| Angolo            | 9       | **81**             |

> **Sposta le pedine verso la periferia e, in particolare, verso gli angoli.** Un angolo vale 81 volte più del centro. Una pedina in angolo minaccia metà scacchiera con le sue catture e non può essere raggiunta da mosse di movimento avversarie.

---

### Regola 3 — Privilegia sempre le catture rispetto ai movimenti

L'ordinamento delle mosse posiziona tutte le catture **prima** di qualsiasi mossa non catturante. Questa scelta non è solo algoritmica, ma riflette un principio strategico:

> **Se hai una cattura disponibile, esplorala prima di qualsiasi spostamento.** Le catture eliminano materiale avversario (il criterio più pesante), aprono linee di attacco e, nella maggior parte dei casi, sono mosse localmente dominanti.

---

### Regola 4 — Elimina prima le pedine avversarie pericolose

La componente P4 (`_capture_dangerous_piece_bonus_from_moves`) identifica le pedine nemiche che hanno almeno una cattura disponibile e assegna un bonus di +12 alla loro eliminazione.

> **Quando puoi scegliere chi catturare, dai priorità alle pedine avversarie che hanno a loro volta catture disponibili.** Eliminare una minaccia immediata evita che l'avversario esegua la sua cattura al turno successivo, guadagnando un doppio vantaggio: si riduce il materiale nemico e si neutralizza un attacco imminente.

---

### Regola 5 — Cattura verso la periferia, non verso il centro

Le componenti P1 (`_capture_outer_bonus`) e l'ordinamento delle catture privilegiano destinazioni di livello assoluto più alto. Una cattura che porta la pedina catturante verso un angolo vale più di una cattura che la riporta al centro.

> **Quando hai più catture possibili, preferisci quella che ti lascia in una posizione più periferica.** Catturare "verso l'esterno" cumula il vantaggio materiale (una pedina avversaria eliminata) con un miglioramento posizionale immediato.

---

### Regola 6 — Muoviti per aprire catture verso le celle più alte

La componente P5 (`_corner_setup_bonus_limited`) valuta le mosse non catturanti guardando un passo avanti: premia i movimenti che, dopo essere stati eseguiti, aprono nuove catture verso celle di livello ≥ 8.

> **Se non hai catture disponibili, muoviti nella posizione che massimizza le catture verso angoli e bordi nel turno successivo.** Ogni mossa non catturante è un investimento: valgono di più le mosse che "caricano" offensive verso la periferia.

---

### Regola 7 — Mantieni la mobilità e la pressione tattica

Le componenti di mobilità (`_W_MOBILITY = 9`) e conteggio catture (`_W_CAPTURE_COUNT = 2`) premiano avere più opzioni dell'avversario, anche quando non si sta catturando.

> **Evita posizioni in cui le tue pedine sono bloccate o hanno poche mosse legali.** La mobilità superiore garantisce flessibilità tattica e forza l'avversario a subire la tua iniziativa; avere più catture disponibili dell'avversario è un segnale diretto di pressione offensiva superiore.

---

### Riepilogo gerarchico delle priorità

```
1. Non perdere pedine          (peso 80 — domina tutto)
2. Occupare angoli/periferia   (peso quadratico sul livello)
3. Catturare prima di muoversi (ordinamento mosse)
4. Eliminare pedine pericolose (bonus +12 per minacce attive)
5. Catturare verso l'esterno   (P1 + ordinamento catture per dst_level)
6. Prepararsi per catture alte (P5 — look-ahead di 1 mossa)
7. Mantenere mobilità          (peso 9 sulla differenza mosse)
```

---

## 3. Architettura del giocatore

Il giocatore è strutturato in quattro strati logici:

```
playerStrategy()          ← entry point: iterative deepening + gestione timeout
    └── _alphabeta()      ← minimax con alpha-beta pruning
            ├── order_moves()       ← ordinamento mosse per efficienza
            └── evaluate_state()   ← valutazione euristica dello stato
                    ├── _positional_value()
                    ├── _capture_outer_bonus()
                    ├── _capture_threat_score_from_caps()
                    ├── _capture_dangerous_piece_bonus_from_moves()
                    └── _corner_setup_bonus_limited()
```

---

## 4. Sistema dei livelli di distanza

La scacchiera è organizzata in **livelli concentrici** calcolati dalla distanza euclidea dal centro. Su una scacchiera 8×8 i livelli vanno da 1 (centro) a 9 (angoli):

```
9 8 7 6 6 7 8 9
8 6 5 4 4 5 6 8
7 5 3 2 2 3 5 7
6 4 2 1 1 2 4 6
6 4 2 1 1 2 4 6
7 5 3 2 2 3 5 7
8 6 5 4 4 5 6 8
9 8 7 6 6 7 8 9
```

Le celle agli **angoli** (livello 9) sono le posizioni più preziose: da lì si può catturare verso quasi tutto il resto della scacchiera, mantenendo una posizione periferica sicura.

Le funzioni helper che leggono questa struttura sono:

```python
def _level(game, r, c):
    return game.distance_levels[r][c]

def _max_level(game):
    return game.distance_levels[0][0]  # Su 8×8 restituisce 9
```

---

## 5. Funzione di valutazione euristica

`evaluate_state(game, state, root_player)` è il cuore del giocatore. Viene chiamata su ogni nodo foglia dell'albero di ricerca (quando si raggiunge la profondità massima o uno stato terminale).

### Casi terminali
Se lo stato è terminale (vittoria o sconfitta), restituisce valori estremi:
```python
if winner == root_player:  return  100_000
if winner is not None:     return -100_000
```

### Punteggio composito
In tutti gli altri casi calcola una somma pesata di sei componenti:

```python
score = (
    _W_PIECES            * (root_pieces - opp_pieces)         # differenza pedine
  + _W_MOBILITY          * (len(root_moves) - len(opp_moves)) # differenza mobilità
  + _W_CAPTURE_COUNT     * (len(root_caps)  - len(opp_caps))  # differenza catture disponibili
  + _W_POSITION          * positional                          # valore posizionale assoluto
  + _W_CAPTURE_OUTER     * cap_outer                           # P1: catture verso periferia
  + _W_THREAT_PRESSURE   * threat_pressure                     # P3: qualità delle catture
  + _W_CAPTURE_DANGEROUS * capture_dangerous                   # P4: cattura pedine pericolose
  + _W_CORNER_SETUP      * corner_setup                        # P5: setup verso angoli
)
```

---

## 6. Componenti euristiche nel dettaglio

### 6.1 Differenza pedine (`_W_PIECES = 80`)

```python
root_pieces - opp_pieces
```

Il componente con peso maggiore. Avere più pedine dell'avversario è il criterio primario di vantaggio materiale. Il peso elevato (80) garantisce che la conservazione delle pedine sia sempre prioritaria rispetto alle considerazioni posizionali.

---

### 6.2 Mobilità (`_W_MOBILITY = 9`)

```python
len(root_moves) - len(opp_moves)
```

Avere più mosse legali a disposizione è un segnale di controllo della scacchiera. Il peso (9) riflette l'importanza della flessibilità tattica.

---

### 6.3 Conteggio catture disponibili (`_W_CAPTURE_COUNT = 2`)

```python
len(root_caps) - len(opp_caps)
```

Distingue la mobilità generica dal potenziale offensivo immediato. Avere più catture disponibili dell'avversario indica pressione tattica superiore.

---

### 6.4 Valore posizionale assoluto — `_positional_value` (`_W_POSITION = 6`)

```python
def _positional_value(game, state, player):
    for r, c in tutte_le_celle:
        lv = _level(game, r, c)
        if cella == player:
            player_val += lv * lv   # livello²
        else:
            opp_val += lv * lv
    return player_val - opp_val
```

**Idea chiave:** il valore di una cella cresce con il **quadrato** del suo livello, non linearmente. Questo rende la differenza tra livello 8 e livello 9 molto più significativa della differenza tra livello 1 e livello 2:

| Livello | Valore lineare | Valore quadratico |
|---------|---------------|-------------------|
| 1       | 1             | 1                 |
| 5       | 5             | 25                |
| 8       | 8             | 64                |
| 9       | 9             | **81**            |

Questo è il **fix principale** rispetto alla v1 (vedi Sezione 10).

---

### 6.5 P1 — Catture verso livelli esterni — `_capture_outer_bonus` (`_W_CAPTURE_OUTER = 1`)

```python
def _capture_outer_bonus(game, captures):
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in captures)
```

Premia le catture che atterrano su celle di livello alto (periferiche). Una cattura verso un angolo (livello 9) vale più di una cattura verso il centro. La differenza tra i bonus di noi e dell'avversario entra nella valutazione.

---

### 6.6 P3 — Pressione tattica — `_capture_threat_score_from_caps` (`_W_THREAT_PRESSURE = 1`)

```python
def _capture_threat_score_from_caps(game, captures):
    score = 0
    for (fr, fc), (tr, tc), _ in captures:
        src_level = _level(game, fr, fc)
        dst_level = _level(game, tr, tc)
        score += 3 * src_level + (max_level - dst_level + 1)
    return score
```

Valuta la **qualità** delle catture disponibili secondo due criteri combinati:
- `3 * src_level`: premia le catture eseguite da pedine già in posizione periferica (più pericolose per l'avversario).
- `max_level - dst_level + 1`: premia le catture verso pedine avversarie più vicine al centro (generalmente più vulnerabili e meno mobili).

---

### 6.7 P4 — Cattura pedine pericolose — `_capture_dangerous_piece_bonus_from_moves` (`_W_CAPTURE_DANGEROUS = 1`)

```python
def _capture_dangerous_piece_bonus_from_moves(game, captures, opponent_moves):
    threatening_pieces = {
        (fr, fc) for (fr, fc), _, is_cap in opponent_moves if is_cap
    }
    bonus = 0
    for (fr, fc), (tr, tc), _ in captures:
        target_level = _level(game, tr, tc)
        bonus += 2 * target_level
        if (tr, tc) in threatening_pieces:
            bonus += 12      # bonus extra per eliminare una pedina minacciante
    return bonus
```

Identifica le **pedine avversarie pericolose**, ovvero quelle che hanno almeno una cattura disponibile nel turno corrente. Catturare una pedina pericolosa dà un bonus fisso di +12, prioritizzando l'eliminazione delle minacce immediate. Il bonus base `2 * target_level` premia comunque la cattura di pedine in posizione periferica.

---

### 6.8 P5 — Setup verso angoli — `_corner_setup_bonus_limited` (`_W_CORNER_SETUP = 4`)

```python
def _corner_setup_bonus_limited(game, state, player, non_captures):
    threshold = max_level - 1   # livello 8 su una 8×8

    candidates = sorted(non_captures,
                         key=lambda m: _level(game, m[1][0], m[1][1]),
                         reverse=True)[:8]

    bonus = 0
    for move in candidates:
        child = game.result(state, move)
        new_caps = catture_disponibili_dopo_move(child, player)
        bonus += count(catture che atterrano su livello >= threshold)
    return bonus
```

Questa euristica guarda **un passo avanti** rispetto alle mosse non catturanti: tra le 8 migliori mosse di posizionamento (ordinate per livello assoluto di destinazione), simula l'effetto di ciascuna e conta quante nuove catture verso livelli alti (≥ 8) diventerebbero disponibili. Premia i movimenti che aprono opportunità offensive verso la periferia.

> **Nota:** i candidati sono limitati a 8 per contenere il costo computazionale, dato che questa funzione chiama `game.result()` per ciascuno.

---

## 7. Ordinamento delle mosse

`order_moves(game, moves)` ordina le mosse prima di esplorarle nell'alpha-beta. Un buon ordinamento massimizza i tagli e riduce drasticamente il numero di nodi visitati.

```python
def move_priority(move):
    (fr, fc), (tr, tc), is_capture = move
    src_level = _level(game, fr, fc)
    dst_level = _level(game, tr, tc)
    delta     = dst_level - src_level

    if is_capture:
        return (0, -dst_level, -src_level)   # catture prima, poi per livello assoluto

    return (1, -dst_level, -delta)            # non-catture dopo, livello assoluto primario
```

**Priorità di ordinamento:**

1. **Catture prima delle non-catture** — le catture tendono ad essere le mosse più forti e devono essere esplorate per prime per generare tagli beta.
2. **Tra le catture:** livello assoluto di destinazione decrescente — catture verso celle più periferiche prima.
3. **Tra le non-catture:** livello assoluto di destinazione come criterio primario, poi delta (differenza di livello) come secondario.

Il criterio 3 è il **fix principale** della v2 (vedi Sezione 10).

---

## 8. Algoritmo Alpha-Beta con gestione del timeout

`_alphabeta(game, state, depth, alpha, beta, maximizing, root_player, deadline)` implementa il classico minimax con potatura alpha-beta.

### Condizioni di terminazione del ramo
```python
if time.perf_counter() >= deadline:
    raise _Timeout()          # timeout: interrompe l'intera ricerca al livello corrente

if game.is_terminal(state):
    return evaluate_state(...)

if depth == 0:
    return evaluate_state(...)
```

### Gestione del passaggio turno
Quando un giocatore non ha mosse legali, il turno viene passato automaticamente senza consumare profondità aggiuntiva significativa:
```python
if not legal_moves:
    passed_state = game.pass_turn(state)
    return _alphabeta(game, passed_state, depth - 1, ...)
```

### Nodi pareggio (mosse con valore uguale)
La lista `best_moves` raccoglie tutte le mosse con il miglior valore trovato. Alla fine viene restituita la prima:
```python
if child_value > value:
    value = child_value
    best_moves = [move]
elif child_value == value:
    best_moves.append(move)   # mantiene tutte le mosse equivalenti
```

### Potatura Alpha-Beta
Il classico meccanismo di potatura è implementato correttamente:
```python
# Nel nodo massimizzante:
alpha = max(alpha, value)
if alpha >= beta:
    break   # taglio beta

# Nel nodo minimizzante:
beta = min(beta, value)
if alpha >= beta:
    break   # taglio alpha
```

---

## 9. Iterative Deepening

`playerStrategy` esplora l'albero di gioco con la tecnica dell'**iterative deepening**: parte dalla profondità 1 e aumenta di 1 a ogni iterazione, fino a quando il tempo a disposizione si esaurisce.

```python
def playerStrategy(game, state, timeout=3):
    legal_moves = game.actions(state)
    if not legal_moves:
        return None

    deadline  = time.perf_counter() + timeout - _TIME_MARGIN
    best_move = random.choice(legal_moves)   # fallback sicuro
    depth     = 1

    while True:
        if time.perf_counter() >= deadline:
            break
        try:
            value, move = _alphabeta(game, state, depth, -inf, +inf,
                                     True, state.to_move, deadline)
            if move is not None:
                best_move = move
            depth += 1
        except _Timeout:
            break

    return best_move
```

**Vantaggi dell'iterative deepening:**
- La mossa `best_move` è sempre aggiornata con il risultato dell'ultima ricerca **completata**: se il tempo scade a metà della profondità `d`, si restituisce la mossa migliore trovata alla profondità `d-1` (che è completa e affidabile).
- Le iterazioni a profondità minore sono molto veloci rispetto all'ultima e costano relativamente poco del budget totale.
- Il fallback iniziale `random.choice(legal_moves)` garantisce che venga sempre restituita una mossa valida, anche se il timeout è brevissimo.

### Margine di sicurezza
```python
_TIME_MARGIN = 0.15   # secondi sottratti al timeout
deadline = time.perf_counter() + timeout - _TIME_MARGIN
```
I 150 ms di margine assicurano che il giocatore non superi mai il limite di tempo imposto dal motore di gioco, lasciando spazio per le operazioni di I/O e overhead del sistema.

---

## 10. Il fix principale rispetto alla v1

### Il bug della v1

Nella versione precedente, l'ordinamento delle mosse non catturanti usava il **delta di livello** (`dst - src`) come criterio primario. Questo causava un comportamento errato in scenari come il seguente:

- La nostra pedina è al livello 6.
- Cella A (angolo) ha livello 9 → delta = +3
- Cella B (libera dopo una cattura avversaria) ha livello 8 → delta = +2

In apparenza, l'angolo (delta=+3) veniva preferito correttamente. Ma il bug emergeva con questo scenario:

- La nostra pedina è al livello 6.
- Cella A ha livello 8 → delta = +2
- Cella B ha livello 8 → delta = +2 ma con `src` diverso

In questi casi di parità del delta, l'euristica non distingueva il **livello assoluto di arrivo**, potendo scegliere la cella sbagliata.

### La soluzione nella v2

Il criterio primario diventa il **livello assoluto di destinazione**. Il delta rimane solo come tiebreaker secondario:

```python
# v1 (bug):
return (1, -delta, -dst_level)

# v2 (fix):
return (1, -dst_level, -delta)
```

Lo stesso principio è applicato alla funzione `_positional_value`, che ora usa `lv * lv` (quadratico) invece di `lv` (lineare), amplificando la differenza tra livelli alti e garantendo che la preferenza per gli angoli sia netta e univoca.

---

## 11. Costanti e pesi

| Costante                | Valore | Componente                                      |
|-------------------------|--------|-------------------------------------------------|
| `_TIME_MARGIN`          | 0.15   | Margine di sicurezza sul timeout (secondi)      |
| `_W_PIECES`             | 80     | Differenza pedine residue                       |
| `_W_MOBILITY`           | 9      | Differenza mosse legali disponibili             |
| `_W_CAPTURE_COUNT`      | 2      | Differenza numero catture disponibili           |
| `_W_POSITION`           | 6      | Valore posizionale assoluto (livello²)          |
| `_W_CAPTURE_OUTER`      | 1      | P1: catture verso celle più esterne             |
| `_W_THREAT_PRESSURE`    | 1      | P3: qualità delle catture disponibili           |
| `_W_CAPTURE_DANGEROUS`  | 1      | P4: cattura pedine pericolose                   |
| `_W_CORNER_SETUP`       | 4      | P5: setup verso angoli/periferia                |

La gerarchia dei pesi riflette le priorità strategiche: il materiale (`_W_PIECES = 80`) domina di gran lunga le considerazioni posizionali, garantendo che il giocatore non sacrifichi pedine in cambio di miglioramenti posizionali marginali.

---

## 12. Entry point — `playerStrategy`

```python
def playerStrategy(game, state, timeout=3) -> move | None
```

**Parametri:**
- `game`: istanza di `ZolaGame`, fornisce le regole e la struttura della scacchiera.
- `state`: istanza di `Board`, stato corrente della partita.
- `timeout`: tempo massimo in secondi (default 3).

**Ritorna:** la mossa migliore trovata nel formato `((fr, fc), (tr, tc), is_capture)`, oppure `None` se non ci sono mosse legali.

---

## 13. Flusso completo di esecuzione

```
playerStrategy(game, state, timeout)
│
├─ Calcola deadline = now + timeout - 0.15
├─ Inizializza best_move = mossa casuale (fallback)
│
└─ Loop iterative deepening (depth = 1, 2, 3, ...):
    │
    └─ _alphabeta(depth, alpha=-∞, beta=+∞, maximizing=True)
        │
        ├─ Timeout? → raise _Timeout → break del loop esterno
        ├─ Terminale o depth==0? → evaluate_state()
        │     ├─ Vittoria/Sconfitta → ±100.000
        │     └─ Somma pesata di 8 componenti euristiche
        │
        ├─ Nessuna mossa legale? → pass_turn + ricorsione
        │
        └─ Per ogni mossa (ordinate da order_moves):
            ├─ game.result(state, move) → stato figlio
            ├─ Ricorsione su figlio (depth-1, not maximizing)
            ├─ Aggiorna alpha/beta
            └─ Potatura se alpha >= beta → break
```

---

*Documentazione per `playerPRR.py`, attualmente lo stato dell'arte del gruppo per il gioco Zola*