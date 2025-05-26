# ASTRA-Pipeline Architecture

```mermaid
flowchart LR
    subgraph CLI
        CLI["CLI Entry\n(src/cli.py)"]
    end

    subgraph Graph_Definition
        G["build_graph\n(src/graph.py)"]
    end

    subgraph Pipeline
        C["collector\n(src/agents/collector.py)"]
        LI["location_inference\n(src/agents/location_inference.py)"]
        F["filter\n(src/agents/filter.py)"]
        S5["sentiment5\n(src/agents/sentiment5.py)"]
        S3["sentiment3\n(src/agents/sentiment3.py)"]
        T["topics\n(src/agents/topics.py)"]
        M["merge\n(src/agents/merge.py)"]
        R["report\n(src/agents/reporter.py)"]
        C --> LI --> F --> S5 --> S3 --> T --> M --> R
    end

    CLI --> G --> C

    subgraph LLM_Abstraction
        LLM["LLM Abstraction\n(src/llm_abstraction.py)"]
    end

    C --> LLM
    LI --> LLM
    S5 --> LLM
    S3 --> LLM
    T --> LLM
    R --> R["Outputs\n(data/, reports/)"]
```

This diagram illustrates the flow from CLI entry, through graph construction, each agent stage, LLM abstraction interactions, and final outputs.
