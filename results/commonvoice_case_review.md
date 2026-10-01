# Cross-speaker listening review (exploratory)

Source: six distinct `client_id` entries from the Common Voice Beijing Chinese `test.tsv` subset. The three cases below were chosen after inspecting the 450-row result CSV, so they are diagnostic examples rather than an unbiased test-set estimate. CER values use Traditional-to-Simplified and Arabic-numeral normalization. No model weights or Q75 thresholds were changed.

| Clip | Added noise / input SNR | Full: output SNR, STOI, CER | 12 dB mix: output SNR, STOI, CER | 6 dB mix: output SNR, STOI, CER | Listening observation |
| --- | --- | --- | --- | --- | --- |
| `18757799` | white / -5 dB | 11.78 dB, 0.83, 0.94 | 6.02 dB, 0.81, 0.61 | 0.82 dB, 0.78, 0.39 | Full sounded muffled and uncomfortable. |
| `18977976` | low-frequency / -5 dB | 7.05 dB, 0.84, 0.65 | 4.13 dB, 0.80, 0.25 | 0.25 dB, 0.74, 0.20 | In Full, the ending “电感的电路元件” was nearly unintelligible; both mixes preserved it better. |
| `22052863` | white / +5 dB | 18.82 dB, 0.91, 0.31 | 15.14 dB, 0.91, 0.15 | 10.60 dB, 0.90, 0.15 | Full retained fewer audible phonetic details and sounded muffled; weaker processing preserved more. |

For all three cases, Q75 selected `none`, so `adaptive_mix` equals Full. These observations support a speech-preservation failure hypothesis for these cases, not a proven cause inside the ERB or DF branch. Full can score higher on SNR and even STOI while a listener and the chosen ASR model find certain words less clear. The short-clip CER and subjective judgments should not be treated as universal performance claims.

Next protocol: develop any revised policy using separate development utterances/speakers, freeze it, then evaluate on untouched test speakers. Because these three test examples were inspected, they should be kept as diagnostic cases, not used to claim unbiased gains from a rule tuned to them.
