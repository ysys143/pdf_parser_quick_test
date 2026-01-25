## **TV-RAG: A Temporal-aware and Semantic Entropy-Weighted** **Framework for Long Video Retrieval and Understanding**



Zongsheng Cao [∗]

agiczsr@gmail.com
Researcher


Feng Chen
chenfeng@lenovo.com
PCIE

**Abstract**



Yangfan He [∗]

he00577@umn.edu
UMN


Zepeng Wang
wangzpb@lenovo.com
PCIE



Anran Liu [∗†]

anniegogo1008@gmail.com
Researcher

Jun Xie [†]

xiejun@lenovo.com
PCIE



Large Video Language Models (LVLMs) have rapidly emerged as
the focus of multimedia AI research. Nonetheless, when confronted
with lengthy videos, these models struggle: their temporal windows are narrow, and they fail to notice fine-grained semantic
shifts that unfold over extended durations. Moreover, mainstream
text-based retrieval pipelines, which rely chiefly on surface-level
lexical overlap, ignore the rich temporal interdependence among
visual, audio, and subtitle channels. To mitigate these limitations,
we propose TV-RAG, a training-free architecture that couples temporal alignment with entropy-guided semantics to improve longvideo reasoning. The framework contributes two main mechanisms:
_(i)_ a time-decay retrieval module that injects explicit temporal offsets into the similarity computation, thereby ranking text queries
according to their true multimedia context; and _(ii)_ an entropyweighted key-frame sampler that selects evenly spaced, informationdense frames, reducing redundancy while preserving representativeness. By weaving these temporal and semantic signals together,
TV-RAG realises a dual-level reasoning routine that can be grafted
onto any LVLM without re-training or fine-tuning. The resulting
system offers a lightweight, budget-friendly upgrade path and consistently surpasses most leading baselines across established longvideo benchmarks such as Video-MME, MLVU, and LongVideoBench,
confirming the effectiveness of our model. The code can be found
at https://github.com/AI-Researcher-Team/TV-RAG.


**CCS Concepts**

- **Computing methodologies** → **Natural language processing** .


**Keywords**

Video Understanding, Temporal-aware, Semantic Entropy-Weighted


∗Equal contribution
†Corresponding author


Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than
the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific
permission and/or a fee. Request permissions from permissions@acm.org.
_MM’25, October 27–31, 2025, Dublin, Ireland_
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-2035-2/2025/10
[https://doi.org/10.1145/3746027.3755873](https://doi.org/10.1145/3746027.3755873)



**ACM Reference Format:**
Zongsheng Cao, Yangfan He, Anran Liu, Feng Chen, Zepeng Wang, and Jun
Xie. 2025. TV-RAG: A Temporal-aware and Semantic Entropy-Weighted
Framework for Long Video Retrieval and Understanding. In _Proceedings of_
_the 33th ACM International Conference on Multimedia (MM ’25), October 27–_
_31, 2025, Dublin, Ireland._ [ACM, New York, NY, USA, 9 pages. https://doi.org/](https://doi.org/10.1145/3746027.3755873)
[10.1145/3746027.3755873](https://doi.org/10.1145/3746027.3755873)


**1** **Introduction**

Recent breakthroughs in large-scale language modelling have catalysed rapid progress in multimodal research, ultimately leading to
a new class of Large Video–Language Models (LVLMs) [11, 16, 37].
Despite their impressive accuracy on short, clip-length inputs, current LVLMs still face significant obstacles when tasked with analyzing and reasoning over very long videos.
Recent attempts to improve long-horizon reasoning in Large
Video Language Models, exemplified by LongVA [39] and LongLLAVA

[35], have largely focused on widening the context window. LongVA,
for instance, simply scales up the token capacity to ingest more
frames. This brute-force expansion, however, proves brittle when
faced with out-of-distribution footage: the Video-MME benchmark
shows its accuracy drops sharply as additional frames are supplied.
A parallel research thread employs Retrieval-Augmented Generation (RAG) to supplement video queries with externally retrieved
documents [15, 18, 40]. Yet, treating the video stream purely as
text ignores vital visual alignment cues, and the reliance on highcapacity LVLM backbones translates into steep computational costs
and poor adaptability. Consequently, existing RAG pipelines struggle to capture the intertwined temporal and semantic signals required for truly robust long-video understanding.
Based on the considerations above, we focus on the following
question: _Can we develop a new model that can tackle the video_
_RAG task from both temporal and semantic perspectives in a unified_
_training-free framework?_
Motivated by this limitation, several studies [15, 18, 40] advocate
replacing the bulky chain of visual tokens with proxy captions distilled directly from the video stream via off-the-shelf optical character recognitions (OCR), automatic speech recognition (ASR), and
object-detection models. These auxiliary texts are tightly coupled
to the visual scene and inject complementary cues unavailable in
raw pixels. For instance, Long-context LVLMs extend visual tokens
so the model sees every frame, yielding strong temporal awareness
but at the cost of heavy retraining and limited semantic depth. GPTbased agent pipelines first convert videos into textual reports, then


MM’25, October 27–31, 2025, Dublin, Ireland Zongsheng Cao et al.


                                     - To handle semantic drift and maintain temporally coherent
retrieval, we introduce a temporal-window BM25 retriever
that ties lexical relevance to timestamp alignment across
auxiliary text streams, supplying the LVLM with context
that stays synchronized with the evolving video content, all
without modifying model weights.

                                    - Extensive experiments on several well-established benchmarks show that TV-RAG achieves state-of-the-art performance and outperforms other models. It can also be used as
a plug-in for other models.


**2** **Related Work**
**2.1** **Large Language Models for Video**



![](./images/TV-RAG.pdf-1-0.png)

**Figure 1: Advantages of our TV-RAG. TV-RAG provides a**
**temporal-aware, semantic-aware, and training-free pipeline**
**that is easily compatible with any LVLM.**


query a proprietary LLM such as GPT-4o; while semantically powerful, they lose fine-grained temporal cues and depend on closedsource services. Yet, the strategy leaves two key issues unresolved:

**C1** Cross-modal dependencies among images, audio, and subtitles remain implicit, hampering performance on tasks that
demand joint reasoning over long temporal spans;
**C2** The fixed context window still cannot follow subtle semantic drifts that emerge gradually throughout lengthy footage.
To this end, we introduce TV-RAG, a novel framework designed
to enhance the comprehension of long videos in LVLMs without
requiring additional training. Specifically, to address (C1), TV-RAG
employs a semantic entropy-based weighting strategy for key frame
selection to evenly distribute selected frames across time, reducing
redundancy, enhancing representativeness, and prioritizing the most
informative frames. To tackle (C2), it incorporates a temporal windowbased BM25 model that integrates time-aligned auxiliary data, including OCR, ASR, and object detection (DET) outcomes. This approach enhances the relevance of text queries by aligning them
with the temporal context of multimedia content, ensuring that retrieved information is contextually appropriate.
By combining explicit temporal order with entropy-based semantic salience, TV-RAG enables a two-stage reasoning process
that removes noise and filters out irrelevant content from long
videos. The module is lightweight and can be added to any existing LVLM without modifying its weights. This plug-and-play design allows TV-RAG to outperform popular models on benchmark
datasets such as Video-MME, MLVU, and LongVideoBench. We
further evaluate TV-RAG on six widely used open-source LVLM
backbones, confirming its effectiveness.
In summary, our contributions are three-fold:

  - To mitigate frame redundancy and preserve cross-modal semantics in long videos, we propose TV-RAG, including a
semantic-entropy key-frame selector that computes multimodal information gain and enforces uniform temporal coverage, yielding compact yet representative frame sets for
downstream LVLM reasoning.



The surge of large language models has sparked parallel efforts to
craft versatile video–language systems. Early work, Video-ChatGPT

[20], processes frames separately and later fuses their representations via spatial–temporal pooling. VideoChat [10] augments appearance embeddings with on-the-fly textual captions to form a
richer clip encoding. To minimise the gap between image and video
pathways, Video-LLaVA [12] introduces a shared projector that
aligns both outputs within a common language latent space. Its
successor, LLaVA-NeXT-Video [40], adapts the LLaVA-NeXT backbone [15] through targeted video fine-tuning. Despite these advances, retrieval-augmented schemes such as Video-RAG [18] still
fall short in tracking the subtle temporal dependencies and layered
semantics present in lengthy, information-dense footage.


**2.2** **Large Language Models for Long-context**
**Video**

A prominent strand of recent work attempts to enlarge the effective context window so that models can reason over intricate video
narratives. LongVA [39] and Long-LLaVA [35] pursue this goal
by first pre-training language backbones on massive text corpora,
banking on the resulting long-sequence handling skills to transfer to video tasks. In contrast, INTP [27] rearranges the incoming
video tokens without any additional training, stretches the window of an LVLM to ingest far more visual information. Yet, these
solutions confront a familiar trade-off: every extra batch of sampled frames inflates computation while delivering only marginal
gains. Because videos contain significant redundancy, and model
capacity is finite. Under these circumstances, most models cannot
achieve satisfactory performance.


**2.3** **Video Understanding by GPT-based Agent**

A complementary line of work investigates using an LLM as a controller that calls external vision or audio modules to parse longform video data, especially for question–answering tasks [5, 21, 29,
32, 36]. MM-VID [13] advances this idea by explicitly pairing each
frame with its textual caption, whereas VLog [14] first extracts
audio and appearance cues via multimodal pre-training and then
distills them into compact textual surrogates. Agent-style frameworks such as VideoAgent [2], DrVideo [19], and OmAgent [38]
go a step further, issuing adaptive queries to retrieve relevant segments before reasoning over them. Despite their ingenuity, these
pipelines still suffer from two practical drawbacks: (i) iterative tool


TV-RAG: A Temporal-aware and Semantic Entropy-Weighted Framework for Long Video Retrieval and Understanding MM’25, October 27–31, 2025, Dublin, Ireland


**Figure 2: The illustration of our framework TV-RAG. The pipeline begins with query decoupling, where the LVLM rewrites**
**the user question into an explicit evidence-retrieval request, cleanly separating search from reasoning. TV-RAG then executes**
**three tightly coupled steps: (i) a semantic-entropy selector distills high-information frames across visual, OCR, ASR and de-**
**tection streams; (ii) a temporal-decay BM25 retriever ranks auxiliary texts by both lexical relevance and timestamp proximity,**
**ensuring chronological coherence; and (iii) a bi-level reasoning routine drafts and self-verifies the answer using the retrieved**
**evidence. By closing the loop between asking, retrieving, and reasoning without altering LVLM weights, TV-RAG simultane-**
**ously sharpens retrieval precision and elevates answer faithfulness.**



![](./images/TV-RAG.pdf-2-0.png)

invocation inflates inference latency, and (ii) many rely on closedsource models, limiting both efficiency and the ease with which the
community can replicate results using purely open-source stacks.
Moreover, there are also some advanced methods, such as [18];
however, they have flaws in capturing the joint associations of temporal and semantic information. In this way, it is still an open issue
to be addressed.


**3** **Methodology**

We introduce TV-RAG, a novel training-free process designed for
LVLMs that can be seamlessly integrated into any existing LVLM.
As shown in Fig. 2, the process consists of three main phases: **(i) Se-**
**mantic entropy-based information extraction:** After obtaining the query, information is extracted based on semantic entropy
from different sources. **(ii) Temporal decay-enhanced retrieval**
**model:** In order to capture the important temporal information
in the video, the time window mechanism request is introduced
for obtaining the relevant information. **(iii) Context-enhanced**
**reasoning-based response generation:** In this final stage, the
auxiliary text retrieved based on the context-enhanced reasoning
mechanism is integrated with the user’s query and fed into the
LVLM to generate the final output.



**Problem Setup.** Let V be an input video. We then use a frame–
selection unit to extract _𝑁_ representative images F Each frame is
then mapped into a visual embedding via a frozen image encoder,
e.g., CLIP-L [23], yielding F _𝑣_ from F . Finally, the visual tokens F _𝑣_
and a user query Q are supplied to a large video–language model
to generate the answer O:

O = LVLM(F _𝑣,_ Q) _._ (1)

In this way, we complete the RAG process for videos.
**Two-stage Processes** . In this paper, motivated by previous efforts

[12, 18], upon receiving a user’s query regarding a video, the LVLM
follows a two-phase process. First, the LVLM decouples the query
to generate structured retrieval requests, which serve as auxiliary
inputs for later stages. In this phase, the LVLM focuses solely on
processing textual data without access to the video frames. These
requests are then parsed, and if any category does not require data,
it will be marked as NULL (indicating no need for that specific
request). This results in the structured retrieval requests, R = {R _𝑖_ },
which means the request categories, such as detection information.
Then, they are passed on to the next phase. The process can thus
be decoupled into two main stages, as formalized in the following
equations:

R = LVLM(P _,_ Q) _,_ R = {R _𝑖_ } _,_ (2)


MM’25, October 27–31, 2025, Dublin, Ireland Zongsheng Cao et al.



where P is the prompt. In the second phase, these requests, R, along
with the video frames, F _𝒗_, are used to generate the final output:


O = LVLM(F _𝒗,_ Q _,_ R) _._ (3)


In this phase, the LVLM processes the video frames, utilizes the
textual query, and incorporates the generated retrieval requests to
provide a more informed and contextually relevant answer.


**3.1** **Semantic Entropy-based Extraction**

**Auxiliary‐Text Extraction and Retrieval.** Following the blueprint laid out by recent retrieval-augmented systems [15, 18, 40],
we first transcribe the raw video into several complementary text
channels and then fetch the portions most useful for the downstream LVLM. Formally, we create a set R = { R _𝑎𝑠𝑟_ _,_ R _𝑜𝑐𝑟_ _,_ R _𝑑𝑒𝑡_ } _,_
where ( _i_ ) _𝑟_ asR is automatic speech recognition to harvest dialogue
or soundtrack content; ( _ii_ ) _𝑟_ det is salient entities present on screen;
and ( _iii_ ) _𝑟_ ocR is optical character recognition to capture any scene
text.
Longer clips naturally produce voluminous and often redundant
token streams, while the context budget of current open-source
LVLMs remains tight. To avoid overflow, we retrieve only those
snippets that align semantically with the user query. Before retrieval, three modality-specific repositories are built in parallel from
the video itself: an ASR base DasR, an OCR base DocR, and an
object-detection base Ddet. Subsequent look-ups are executed against
these lightweight databases, ensuring that irrelevant tokens do not
enter the LVLM’s limited context window.
**Building Retrieval Repositories.** Open source LVLMs tend to
misread scene text and spoken words, falling short of their proprietary counterparts. To curb such hallucinations and exploit frame
content more effectively, we offload text extraction to specialist
models. Concretely, EasyOCR [7] is run on each key frame to harvest on-screen captions, giving a pool of strings TocR; meanwhile,
the soundtrack is transcribed by Whisper [24], yielding an ASR
transcript TasR as advocated in prior work [15, 18, 40]. Both text
streams are embedded with the ContRieveR encoder [6] to obtain dense vectors, which are written to two separate FAISS indices [8]: DocR for scene text and DasR for speech. This design enables low-latency, similarity-based retrieval of the most relevant
snippets during query time.
**Key–Frame Selection.** Although modern LVLMs excel at recognising objects, they remain error-prone when asked to count instances, pinpoint locations, or reason about complex interactions,
frequently hallucinating details when contextual cues are sparse.
To curb this issue, motivated by [12, 18], we rank every sampled
frame F = { _𝐹𝑡_ } by the semantic affinity between the detector request _𝑅𝑑𝑒𝑡_ and the frame content, modulated by a temporal importance weight:

           -           _𝐹𝑘𝑒𝑦_ = _𝐹𝑡_ �� _𝛼𝑡_        - CLIP� _𝑅𝑑𝑒𝑡_ _, 𝐹𝑡_        - ≥ _𝜏_ _,_


where _𝜏_ is a similarity threshold. The weight _𝛼𝑡_ captures how much
new information the segment contributes and is computed from
the normalised Shannon entropy of its visual features:

_𝛼𝑡_ = ~~�~~ _𝐻_ ( _𝐹𝑡_ )
_𝑗_ _[𝐻]_ [(] _[𝐹]_ _𝑗_ [)] _[.]_



Object detection is then run only on this entropy-aware subset
FKey, ensuring that the LVLM processes the most informative and
contextually relevant frames.
The weight _𝑤𝑡_ assigned to each frame is determined by the _infor-_
_mation entropy 𝐻_ ( _𝐹𝑡_ ), a measure of the uncertainty or variability
within the frame’s content. Then the entropy _𝐻_ ( _𝐹𝑡_ ) is computed
as:

CLIP similarity( _𝑅_ det _, 𝐹𝑡_ )
_𝐻_ ( _𝐹𝑡_ ) = − _𝑝𝑡_ log _𝑝𝑡,_ _𝑝𝑡_ = ~~�~~ (4)
_𝑗_ [CLIP similarity][(] _[𝑅]_ det _[, 𝐹]_ _𝑗_ [)] _[,]_

where _𝑝𝑡_ represents the normalized similarity between the object
retrieval query and each frame _𝐹𝑡_, and _𝐻_ ( _𝐹𝑡_ ) captures how much
variance exists in the content of a given time segment. The selection process begins by computing the CLIP similarity [23] between an object retrieval query R _𝑑𝑒𝑡_ and the sampled video frames
F. High entropy values indicate significant changes in the visual
or auditory content, suggesting that this segment contains crucial,
non-redundant information.


RemaRK. _In this way, it ensures that time segments with higher_
_variability in content are more likely to be selected, while segments_
_with low variability (i.e., those that are redundant) are deprioritized._
_The goal is to maximize the representativeness of the selected frames_
_and minimize redundant or irrelevant content, which is particularly_
_important in the context of long-duration videos. By incorporating_
_this entropy-based weighting mechanism, we ensure that critical mo-_
_ments in the video, which may involve sudden shifts in object inter-_
_action or scene changes, are not overlooked._


**3.2** **Temporal Window-based Text Retrieval**

**Query–Aware Retrieval.** Our pipeline introduces a dedicated retrieval routine that tightly couples the user prompt with the auxiliary captions mined from the video. Inspired by prior multimodal
RAG work [12, 18], we begin by converting the query into a dense
representation e _𝑟𝑒𝑞_ using the Contriever encoder.
To prevent the common temporal drift problem, where a lexically relevant caption is drawn from an incorrect moment in the
clip, we refine the top- _𝑘_ text hits with a temporal re-ranking stage.
Each candidate’s score is updated by a weighted sum of its lexical
similarity to e _𝑟𝑒𝑞_ and its time-stamp distance from the query segment, ensuring the final selection is both semantically pertinent
and temporally on point.
**Temporal Re-scoring of Retrieved Snippets.** Assume that a
FAISS search over the OCR/ASR indices yields the relevant snippets C. Let _𝜏𝑞_ ∈ R mark the moment in the video that anchors the
user’s query, and denote by _𝜏𝑖_ the original time-code of snippet _𝑐𝑖_ .
We refine each snippet’s lexical BM25 score by a time-aware decay:



BM25(ereq _,𝑐𝑖_ ) _𝑒_ [−] [�] _𝑘_ [2] =0 _[𝜆][𝑘]_ [|] _[𝜏][𝑘𝑞]_ [−] _[𝜏][𝑖]_ [|]
_𝑠_ ˜ _𝑖_ =



_,_ (5)

�3 _𝐾_

BM25(ereq _,𝑐_ _𝑗_ ) _𝑒_ [−] [�] _𝑘_ [2] =0 _[𝜆][𝑘]_ [|] _[𝜏][𝑘𝑞]_ [−] _[𝜏]_ _[𝑗]_ [|]
_𝑗_ =1



�3 _𝐾_



where ereq is the Contriever embedding of the query and _𝜆_ _>_ 0
(set to 1 in our study) controls the strength of the temporal penalty.
Here the three anchor points _𝜏𝑞_ _[𝑘]_ [serve complementary roles:] _[ 𝜏]_ _𝑞_ [0] [is]
the time-code of the _last_ video frame, _𝜏𝑞_ [1] [marks the] _[ first]_ [ frame, and]
_𝜏𝑞_ [2] [corresponds to the frame whose local context is semantically]
most similar to the query. The rationale is that salient evidence can


TV-RAG: A Temporal-aware and Semantic Entropy-Weighted Framework for Long Video Retrieval and Understanding MM’25, October 27–31, 2025, Dublin, Ireland



appear near a video’s conclusion (e.g., a reveal or punch-line), at
its outset (introductory setup), or at an unpredictable location that
is best located via semantic affinity. By injecting all three anchors
into the exponential decay, the rescoring scheme remains agnostic
to where the key information actually lies, yet still rewards snippets clustered around whichever temporal region proves relevant
for the user’s request.
The exponential term softly favours snippets that occur near _𝜏𝑞_ _[𝑘]_
while still allowing distant, but lexically pertinent candidates to
survive with a lower confidence. Finally, we keep the top- _𝐾_ snippets with the highest ˜ _𝑠𝑖_ :


                            C = TopK [�] C _,_ ˜ _𝑠𝑖_ _._ (6)


Incorporating timestamp proximity into the similarity score remedies two chronic weaknesses of standard retrieval pipelines. (i) It
sharply reduces the likelihood of selecting passages that look semantically relevant yet come from the wrong moment in the footage,
a pitfall common to long videos with recurrent motifs. (ii) It maintains chronological consistency for the LVLM, so downstream reasoning and text generation remain anchored to the correct slice
of the narrative. Unlike hard sliding windows or ad-hoc alignment
rules, the proposed weighting scheme is continuous, differentiable,
and plug-compatible with existing multimodal retrieval frameworks.
**Object Information Retrieval.** After retrieving the relevant information A, the next step involves extracting _object-related infor-_
_mation_ stored in the auxiliary database. Motivated by [15, 18, 40],
object detection models typically produce raw outputs that include
spatial attributes in the form of bounding boxes and other information. These components collectively form a structured scene graph,
expressed as Ast, which can be regarded as a detection database and
enhances the ability of LVLMs to interpret temporal and spatial information. By incorporating this structured graph representation,
we enable LVLMs to reason over the spatial layout and temporal
dynamics of objects within the video.


**3.3** **Context-augmented Reasoning with Query**
**Reformulation**

While retrieved auxiliary signals (e.g., OCR, ASR) can provide essential external context, relying exclusively on these sources introduces two key limitations: (i) they are often noisy, fragmented,
or partially misaligned with the user’s underlying intent; and (ii)
compressing them to fit the limited context window of LVLMs may
introduce semantic loss and degrade the coherence of downstream
reasoning.
To overcome these issues, we propose a context-augmented reasoning strategy that enhances retrieved evidence through a dualchannel mechanism. This approach not only supplements the auxiliary inputs but also enhances the model’s understanding of the
user query via semantic rephrasing.
**Level-1: Evidence-based Contextualization.** We first aggregate
auxiliary textual signals from _𝐴𝑠𝑡_ . This aggregated context reflects
the explicit knowledge retrieved from the video content and acts
as grounded, observable evidence for reasoning.



**Level-2: Query-enhanced Latent Reasoning.** Recognizing that
retrieved evidence may still omit latent or abstract concepts implied by the user’s query, we introduce a generation-based augmentation step. Instead of compressing auxiliary data, we instruct
the LVLM to perform two complementary operations: (1) generate
general background context relevant to the query to enrich modellevel knowledge grounding; (2) reformulate the user query into a
set of semantically similar paraphrases to enhance retrieval sensitivity and decoding diversity.
Specifically, we define the prompt _𝐼𝑔𝑒𝑛_ to guide context generation as:


Given a question, first generate a helpful background
context. Then, provide 2-3 alternative phrasings
of the question with similar meaning.


This produces two outputs: a generated context Cgen and a set of
_𝑘_ semantically equivalent queries {Q [(] _[𝑖]_ [)] } _[𝑘]_ _𝑖_ =1 [. These components en-]
able the model to reason over both retrieved facts and inferred intent.
**Final Input Composition.** The final structured input to the LVLM
integrates the sampled video frames, the retrieved auxiliary evidence, the original query, and the generated latent context:


O = LVLM( _𝐹𝑘𝑒𝑦,_ Concat(A _𝑠𝑡_ _,_ Q _,_ {Q [(] _[𝑖]_ [)] } _[𝑘]_ _𝑖_ =1 _[,]_ [ C][gen][))] _[.]_ (7)


RemaRK. _By combining retrieval-grounded evidence with genera-_
_tive reasoning and query reformulation, this bi-level design strength-_
_ens both explicit and implicit context understanding. It also mitigates_
_alignment gaps between noisy auxiliary signals and the model’s pre-_
_trained knowledge space, resulting in more robust and semantically_
_coherent outputs._


**4** **Experiments**
**4.1** **Datasets**

We gauge the effectiveness of our framework on three publicly
accepted long-video testbeds. Video-MME [4] contains real-world
clips that run anywhere from 11 s to roughly one hour and probe
fine-grained understanding of everyday scenarios. MLVU [42] spans
nine evaluation tasks sourced from videos lasting 3 min to 2 h
(mean length about 12 min), challenging models to reason across
extended temporal ranges. Finally, LongVideoBench [33] targets
multimodal retrieval and reasoning in lengthy footage, offering
6,678 human-written multiple-choice questions distributed over 17
thematic categories.


**4.2** **Experimental Implementation**

All experiments were executed on NVIDIA A100 GPUs. At the first
stage, we prune the LVLM’s detection prompts, keeping only concrete, CLIP-responsive objects and discarding abstract terms. For
following retrieval, we fix both the FAISS and CLIP acceptance
thresholds to 0 _._ 3; FAISS scores are computed with the IndexFlatIP
backend [8]. Our ablation suite is built around Long-LLaVA 7B [35],
whose extended context window and modest footprint make it
ideal for probing how RAG similarity cut-offs and frame-sampling
rates affect performance.


MM’25, October 27–31, 2025, Dublin, Ireland Zongsheng Cao et al.


**Model** **#Text** **LLM Params** **Frames** **Short** **Medium** **Long** **Overall** **Gain**

_**Proprietary LVLMs**_


_**Open-Source LVLMs**_


**Table 1: Performance evaluation on the Video-MME [4] benchmark.The #Text column logs the average volume of tokens that**
**TV–RAG appends for each sample. Most strikingly, the 72 B LLaVA-Video model [41] outfitted with our adapter edges past**
**the commercial Gemini-1.5-Pro [25]. The baseline results are from [18]. All open-source baselines and their TV–RAG variants**
**were re-benchmarked.**



![](./images/TV-RAG.pdf-5-1.png)

**4.3** **Main Results**

**Video-MME** . To conduct a fair test of TV-RAG, we fixed every
candidate LVLM to the same 32-frame input budget, an especially
practical ceiling for the 72B-parameter models and a convenient
checkpoint for their 7B counterparts. Our benchmark therefore
covers six systems: four open 7B backbones (Video-LLaVA [12],
LLaVA-NeXT-Video [40], LongVA [39], Long-LLaVA [35]) and two
72B giants (Qwen2-VL [31], LLaVA-Video [41]). As reported in Table 1, attaching TV-RAG to the 72B pair lifts their scores beyond
the proprietary Gemini-1.5-Pro baseline [25]. Averaged over all six
backbones, the pipeline delivers a significant gain, with the most
dramatic jumps on extended footage. The boost stems from injecting ∼14 additional keyframes (about 2 000 tokens in total, given
the typical 144-token payload per frame), which compensates for
the weak visual grounding of LVLMs that were largely pre-trained
on text. These extra, text-rich frames act as semantic anchors and
enable sharper comprehension of complex video content.
**MLVU** . Table 2 contrasts leading LVLMs on the benchmark’s multiplechoice split. The vanilla 7B LLaVA-Video line scores 70.8, but once
TV-RAG is attached, the figure climbs to 72.6, an absolute gain of
+1 _._ 8 that propels the small model ahead of every open-source rival
under 70B, including the 32B Oryx-1.5 (72.3). A similar but smaller
gain is observed for the 72B backbone, improving from 73.1 to 73.4
and further widening the gap with the proprietary GPT-4o baseline (64.6). These improvements, achieved with the same 64-frame
budget, underline TV-RAG’s ability to supply complementary textual cues that sharpen video understanding, especially for lighter
models that still have headroom to benefit from richer context.
**LongVideoBench** . We next gauge the influence of TV–RAG on
both 7B and 72B LLaVA–Video backbones via the LongVideoBench
suite [33]. For clarity, we adopt the plain 64–frame feed and omit



**Model** **#Params Frames** **Overall**

_**Proprietary LVLMs**_

|Open-Source LVLMs|Col2|
|---|---|
|Video-CCAM [3]<br>14B<br>96<br>Video-XL [28]<br>7B<br>256<br>Aria [9]<br>25.3B<br>256<br>LLaVA-Video* [41]<br>7B<br>64<br>Oryx-1.5 [17]<br>32B<br>128<br>LLaVA-Video* [41]<br>72B<br>64|63.1<br>64.9<br>70.6<br>70.8<br>72.3<br>73.1|
|LLaVA-Video + TV-RAG<br>7B<br>64<br>LLaVA-Video + TV-RAG<br>72B<br>64|72.6<br>**73.4**|



**Table 2: The overall performance in the multiple-choice task**
**of the MLVU [42] benchmark. * donates the results of our**
**replication.**


the benchmark’s optional interleaved prompt. Table 3 reveals that
the 72B configuration, once augmented with our retrieval layer,
climbs to the runner-up position on the validation leaderboard:
its score edges past Gemini-1.5-Pro [25] by 1 _._ 6% and trails GPT4o [22] by merely 1 _._ 1%. The lighter 7B model also profits, posting a
+2 _._ 2% uplift under the same experimental regime. These results underscore TV–RAG’s capacity to boost long-video reasoning across
model scales without altering input structure.


**4.4** **Ablation Studies**

We probed the sensitivity of TV–RAG to the volume of visual evidence by running the Long–LLaVA–7B backbone [35] under four


TV-RAG: A Temporal-aware and Semantic Entropy-Weighted Framework for Long Video Retrieval and Understanding MM’25, October 27–31, 2025, Dublin, Ireland



**Model** **#Params Frames Overall**

_**Proprietary LVLMs**_

|Open-Source LVLMs|Col2|
|---|---|
|VideoChat2-Mistral [10]<br>7B<br>8<br>ShareGPT4Video [1]<br>7B<br>8<br>LLaVA-Next-Mistral [15]<br>7B<br>8<br>PLLaVA [34]<br>34B<br>16<br>LLaVA-Video [41]<br>7B<br>64<br>LLaVA-Video [41]<br>72B<br>64|39.3<br>39.7<br>49.1<br>53.2<br>56.6<br>61.9|
|LLaVA-Video + TV-RAG<br>7B<br>64<br>LLaVA-Video + TV-RAG<br>72B<br>64|58.8<br>**65.6**|



**Table 3: The overall performance on the validation set of**
**LongVideoBench [33].**


**Figure 3: Performance with different sampling frames rate**
**on Video-MME [4] when using Long-LLaVA-7B [35] as the**
**LVLM.**


budgets: 8, 16, 32, and 64 sampled frames. Figure 3 charts the accuracies and reveals two clear trends. First, the retrieval layer delivers a measurable lift at every budget; second, the gap widens as
more frames are fed in, with the biggest jump on lengthy clips. In
the vanilla baseline (no retrieval), performance tops out at the 32frame mark, which demonstrates the effectiveness of our model.
We isolate each textual cue produced by the TV-RAG retriever,
including semantic-entropy weighting (SE), temporal-window gating (TW), OCR strings, ASR transcripts, and the context-enrichment
layer, and feed successive combinations to Long–LLaVA–7B [35].
Performance on Video-MME [4] (Table 4) climbs steadily with every new source, with ASR and OCR delivering the significant jump
across short and long clips alike, confirming the unique value of
ASR and OCR transcripts.
SE (semantic-entropy) and TW (temporal-window) act as quality filters, discarding semantically weak or temporally mismatched
snippets so that only the most relevant text aligns with each frame



|SE TW OCR ASR Context|Short Medium Long|Overall|
|---|---|---|
|×<br>✓<br>✓<br>✓<br>✓<br>✓<br>×<br>✓<br>✓<br>✓<br>✓<br>✓<br>×<br>✓<br>✓<br>✓<br>✓<br>✓<br>×<br>✓<br>✓<br>✓<br>✓<br>✓<br>×<br>✓<br>✓<br>✓<br>✓<br>✓|~~65.3~~<br>~~56.5~~<br>~~49.1~~<br>64.5<br>52.7<br>50.3<br>68.9<br>57.8<br>55.0<br>68.3<br>59.4<br>57.8<br>65.4<br>55.6<br>56.4<br>**72.4**<br>**64.1**<br>**59.3**|~~57.0~~<br>55.8<br>60.6<br>61.8<br>59.1<br>**65.3**|


**Table 4: Results for combinations of different extracted texts**
**in Video-MME using LLaVA-Video-7B as the LVLM.**


sequence. Taken together, these findings affirm the rationale behind TV-RAG: multimodal retrieval, temporal alignment, and adaptive weighting converge to produce more faithful and grounded
video-language reasoning.


**Figure 4: Qualitative result shown in Video-MME [4] bench-**
**mark when applying TV-RAG with LLaVA-Video [41].**


Table 6 reports how the similarity cutoff _𝜏_ shapes both efficiency
and accuracy when Long-LLaVA 7B answers the benchmark’s multiplechoice questions. Lower bars ( _𝜏_ =0–0 _._ 1) flood the prompt with more
auxiliary tokens, pushing the inference time to 30-40 s but only
edge the overall score to 63-64. Raising the bar gradually prunes
redundant text and accelerates decoding; at _𝜏_ =0 _._ 3 the model processes 1.9 K tokens in 13 s yet still logs 61.6 overall, while posting the single best medium-clip accuracy (61.0). Beyond this point,
the gains erode: _𝜏_ =0 _._ 5 and _𝜏_ =1 _._ 0 shrink the token load to ≤ 0 _._ 3
K and time to ≤ 9 s, but overall correctness drops sharply to 56.7
and 52.5, respectively. Hence _𝜏_ =0 _._ 3 offers the most balanced tradeoff—halving latency relative to the lowest cutoffs, staying within
typical context limits ( ∼1.9 K extra tokens), and preserving strong
performance across clip lengths. For LVLMs with tighter windows,
a softer cutoff such as _𝜏_ =0 _._ 2 may be preferred, still keeping the token budget under 3 K while avoiding the steep accuracy loss seen
at stricter thresholds.
To study the effectiveness of our model, we compare it with
Video-RAG [18]. As shown in Table 5, our method demonstrates a
clear performance boost over Video-RAG across all video lengths.
This improvement is observed in both the Long-Video and LLaVAVideo models, with particularly noticeable gains in handling shorter
and longer videos. Overall, our model contributes to better model
performance by effectively addressing the challenges posed by different video durations. The observed gains can be attributed to our
framework’s ability to integrate temporal semantics more adaptively.



![](./images/TV-RAG.pdf-6-1.png)

![](./images/TV-RAG.pdf-6-2.png)
MM’25, October 27–31, 2025, Dublin, Ireland Zongsheng Cao et al.


**Figure 5: Grad-CAM heatmaps of the final hidden state, alongside t-SNE projections of the user’s query and keyframe features,**
**are visualized for the example in Figure 4. The combined visualization makes clear that the supplementary texts retrieved by**
**TV-RAG tighten the vision–language link: they redirect the network’s attention toward the frames most pertinent to the**
**question, thereby boosting both answer precision and contextual fidelity.**



![](./images/TV-RAG.pdf-7-0.png)

|𝜏|#Token Time|Short Medium Long|Overall|
|---|---|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>1.0|3.6K<br>39s<br>3.4K<br>33s<br>2.7K<br>18s<br>1.9K<br>13s<br>0.8K<br>10s<br>0.3K<br>9s<br>0.0K<br>8s|66.3<br>58.4<br>57.1<br>68.7<br>57.3<br>59.1<br>66.4<br>60.2<br>58.2<br>67.4<br>**61.0**<br>58.8<br>64.6<br>58.8<br>58.5<br>63.4<br>54.5<br>50.1<br>60.4<br>51.3<br>44.7|63.4<br>62.9<br>60.5<br>61.6<br>60.3<br>56.7<br>52.5|


**Table 6: Performance with different thresholds of retrieval**
**on Video-MME [4] when using Long-LLaVA 7B [35] as the**
**LVLM. #Token and Time denote the total token number of**
**the extracted texts and the average inference time per ques-**
**tion, respectively.**

|Model|Params Frames|Short Medium Long Overall|
|---|---|---|
|X=LongVA<br>X+Video-RAG<br>X+ TV-RAG|7B<br>32<br>7B<br>32<br>7B<br>32|60.9<br>49.3<br>44.0<br>51.4<br>65.4<br>59.1<br>55.7<br>60.1<br>66.2<br>62.1<br>58.7<br>61.7|
|X=LLaVA-Video<br>X+Video-RAG<br>X+ TV-RAG|72B<br>32<br>72B<br>32<br>72B<br>32|78.0<br>63.7<br>59.6<br>67.1<br>81.1<br>72.9<br>73.1<br>75.7<br>82.3<br>73.1<br>73.8<br>76.5|



**Table 5: Performance with other model on Video-MME.**


**4.5** **More Studies**

We present more expriments for Video-MME [4] in Figure 4 and
Figure 5 to illustrate the effectiveness of our approach. As shown
in the figures, integrating external tools with LLaVA-Video to process and retrieve extracted textual information from videos significantly enhances the model’s ability to mitigate visual hallucinations. This integration leads to more accurate and contextually
grounded responses to user queries by reinforcing the alignment
between visual content and textual understanding.
Furthermore, the Grad-CAM [26] and t-SNE [30] visualization
results provide additional evidence supporting the impact of TVRAG in strengthening cross-modal alignment within the LVLM.



Specifically, the Grad-CAM results highlight improved attention to
semantically relevant regions in the video frames, while the t-SNE
analysis reveals a more structured representation space, indicating
a refined relationship between visual and textual modalities. These
findings collectively underscore the effectiveness of our approach
in enhancing the robustness and interpretability of LVLM-based
video understanding.
In summary, the qualitative examples not only highlight the correctness of predictions but also provide insight into how the model
reasons through retrieved information.


**5** **Conclusion**

We propose TV-RAG, a training-free adaptor that upgrades longvideo reasoning in large vision-language models. The scheme fuses
two key ideas: (i) a sliding temporal window that aligns retrieved
cues with the correct time span and (ii) an entropy-based score
that favours semantically rich segments, together compensating
for the narrow temporal receptive field of standard LVLMs and for
their difficulty in tracing subtle scene-level shifts. Contrary to retrievers that rely purely on text similarity or pipelines that demand
full model fine-tuning, TV-RAG clips onto any existing backbone
as a light, plug-and-play module. Extensive trials on Video-MME,
MLVU, and LongVideoBench confirm marked gains, allowing the
patched models to overtake contemporary leaders such as Gemini1.5-Pro and GPT-4o in some metrics. The future work targets finer
temporal gating and adaptive entropy scaling to further strengthen
cross-modal alignment and deep contextual reasoning. In future
work, we will extend TV-RAG to dynamically adjust its temporal
window size based on content uncertainty and explore curriculumstyle retrieval schedules to better adapt to ultra-long, open-world
video streams. We also plan to generalise the framework to audiovisual co-reasoning and egocentric or 360° footage, broadening its
applicability across emerging multimodal benchmarks.


**6** **Acknowledgments**

This work was supported by the Science and Technology Innovation 2030-Key Project under Grant 2021ZD0201404.


TV-RAG: A Temporal-aware and Semantic Entropy-Weighted Framework for Long Video Retrieval and Understanding MM’25, October 27–31, 2025, Dublin, Ireland



**References**

[1] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. 2024. ShareGPT4Video:
Improving Video Understanding and Generation with Better Captions. _arXiv_
_preprint arXiv:2406.04325_ (2024).

[2] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. 2025.
Videoagent: A memory-augmented multimodal agent for video understanding.
In _European Conference on Computer Vision_ . Springer, 75–92.

[3] Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu, and Hui Wang.
2024. Video-ccam: Enhancing video-language understanding with causal crossattention masks for short and long videos. _arXiv preprint arXiv:2408.14023_
(2024).

[4] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan
Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. 2024. Video-MME:
The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in
Video Analysis. _arXiv preprint arXiv:2405.21075_ (2024).

[5] Tanmay Gupta and Aniruddha Kembhavi. 2023. Visual programming: Compositional visual reasoning without training. In _Proceedings of the IEEE/CVF Confer-_
_ence on Computer Vision and Pattern Recognition_ . 14953–14962.

[6] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense information retrieval with contrastive learning. _arXiv preprint arXiv:2112.09118_
(2021).

[[7] JaidedAI. 2023. EasyOCR. https://github.com/JaidedAI/EasyOCR.](https://github.com/JaidedAI/EasyOCR)

[8] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity
search with GPUs. _IEEE Transactions on Big Data_ 7, 3 (2019), 535–547.

[9] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao
Niu, Guoyin Wang, Bei Chen, and Junnan Li. 2024. Aria: An Open Multimodal
Native Mixture-of-Experts Model. _arXiv preprint arXiv:2410.05993_ (2024).

[10] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang,
Limin Wang, and Yu Qiao. 2023. Videochat: Chat-centric video understanding.
_arXiv preprint arXiv:2305.06355_ (2023).

[11] Yanwei Li, Chengyao Wang, and Jiaya Jia. 2025. Llama-vid: An image is worth
2 tokens in large language models. In _European Conference on Computer Vision_ .
Springer, 323–340.

[12] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. 2023. Video-llava:
Learning united visual representation by alignment before projection. _arXiv_
_preprint arXiv:2311.10122_ (2023).

[13] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab,
Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al. 2023.
Mm-vid: Advancing video understanding with gpt-4v (ision). _arXiv preprint_
_arXiv:2310.19773_ (2023).

[14] Qinghong Lin. 2023. VLog: Transform Video as a Document with ChatGPT,
[CLIP, BLIP2, GRIT, Whisper, LangChain. https://github.com/showlab/VLog.](https://github.com/showlab/VLog)

[15] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and
Yong Jae Lee. 2024. LLaVA-NeXT: Improved reasoning, OCR, and world knowl[edge. https://llava-vl.github.io/blog/2024-01-30-llava-next/](https://llava-vl.github.io/blog/2024-01-30-llava-next/)

[16] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao.
2024. Oryx MLLM: On-Demand Spatial-Temporal Understanding at Arbitrary
Resolution. _arXiv preprint arXiv:2409.12961_ (2024).

[17] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao.
2024. Oryx MLLM: On-Demand Spatial-Temporal Understanding at Arbitrary
Resolution. _arXiv preprint arXiv:2409.12961_ (2024).

[18] Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang,
Jiayi Ji, Fei Chao, Jiebo Luo, and Rongrong Ji. 2024. Video-RAG: Visuallyaligned Retrieval-Augmented Long Video Comprehension. _arXiv preprint_
_arXiv:2411.13093_ (2024).

[19] Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li, Hamid Rezatofighi,
and Jianfei Cai. 2024. DrVideo: Document Retrieval Based Long Video Understanding. _arXiv preprint arXiv:2406.12846_ (2024).

[20] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan.
2023. Video-chatgpt: Towards detailed video understanding via large vision and
language models. _arXiv preprint arXiv:2306.05424_ (2023).

[21] Juhong Min, Shyamal Buch, Arsha Nagrani, Minsu Cho, and Cordelia Schmid.
2024. MoReVQA: Exploring Modular Reasoning Models for Video Question Answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pat-_
_tern Recognition_ . 13235–13245.




[[22] OpenAI. 2024. GPT-4o System Card. https://openai.com/index/gpt-4o-system-](https://openai.com/index/gpt-4o-system-card/)
[card/.](https://openai.com/index/gpt-4o-system-card/)

[23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al.
2021. Learning transferable visual models from natural language supervision. In
_International conference on machine learning_ . PMLR, 8748–8763.

[24] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey,
and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In _International conference on machine learning_ . PMLR, 28492–28518.

[25] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. 2024. Gemini 1.5: Unlocking multimodal understanding
across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ (2024).

[26] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. 2017. Grad-cam: Visual explanations from
deep networks via gradient-based localization. In _Proceedings of the IEEE inter-_
_national conference on computer vision_ . 618–626.

[27] Yuzhang Shang, Bingxin Xu, Weitai Kang, Mu Cai, Yuheng Li, Zehao Wen, Zhen
Dong, Kurt Keutzer, Yong Jae Lee, and Yan Yan. 2024. Interpolating Video-LLMs:
Toward Longer-sequence LMMs in a Training-free Manner. _arXiv preprint_
_arXiv:2409.12963_ (2024).

[28] Yan Shu, Peitian Zhang, Zheng Liu, Minghao Qin, Junjie Zhou, Tiejun Huang,
and Bo Zhao. 2024. Video-XL: Extra-Long Vision Language Model for HourScale Video Understanding. _arXiv preprint arXiv:2409.14485_ (2024).

[29] Dídac Surís, Sachit Menon, and Carl Vondrick. 2023. Vipergpt: Visual inference
via python execution for reasoning. In _Proceedings of the IEEE/CVF International_
_Conference on Computer Vision_ . 11888–11898.

[30] Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using tSNE. _Journal of machine learning research_ 9, 11 (2008).

[31] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin
Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024. Qwen2-VL: Enhancing
Vision-Language Model’s Perception of the World at Any Resolution. _arXiv_
_preprint arXiv:2409.12191_ (2024).

[32] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. 2024. Videoagent: Long-form video understanding with large language model as agent. _arXiv_
_preprint arXiv:2403.10517_ (2024).

[33] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. 2024. Longvideobench: A
benchmark for long-context interleaved video-language understanding. _arXiv_
_preprint arXiv:2407.15754_ (2024).

[34] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng.
2024. Pllava: Parameter-free llava extension from images to videos for video
dense captioning. _arXiv preprint arXiv:2404.16994_ (2024).

[35] Yin Song and Chen Wu and Eden Duthie. 2024. aws-prototyping/long-llava[qwen2-7b. https://huggingface.co/aws-prototyping/long-llava-qwen2-7b](https://huggingface.co/aws-prototyping/long-llava-qwen2-7b)

[36] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit
Bansal, and Gedas Bertasius. 2023. A simple llm framework for long-range video
question-answering. _arXiv preprint arXiv:2312.17235_ (2023).

[37] Hang Zhang, Xin Li, and Lidong Bing. 2023. Video-llama: An instructiontuned audio-visual language model for video understanding. _arXiv preprint_
_arXiv:2306.02858_ (2023).

[38] Lu Zhang, Tiancheng Zhao, Heting Ying, Yibo Ma, and Kyusong Lee. 2024. Omagent: A multi-modal agent framework for complex video understanding with
task divide-and-conquer. _arXiv preprint arXiv:2406.16620_ (2024).

[39] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. 2024. Long
context transfer from language to vision. _arXiv preprint arXiv:2406.16852_ (2024).

[40] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi
Feng, Ziwei Liu, and Chunyuan Li. 2024. LLaVA-NeXT: A Strong Zero-shot
Video Understanding Model. [https://llava-vl.github.io/blog/2024-04-30-llava-](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/)
[next-video/](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/)

[41] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and
Chunyuan Li. 2024. Video Instruction Tuning With Synthetic Data.
[arXiv:2410.02713 [cs.CV] https://arxiv.org/abs/2410.02713](https://arxiv.org/abs/2410.02713)

[42] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong,
Bo Zhang, Tiejun Huang, and Zheng Liu. 2024. MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding. _arXiv preprint arXiv:2406.04264_
(2024).


