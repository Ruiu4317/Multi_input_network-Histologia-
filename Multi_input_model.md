graph TD
    A[Input Images\n(Different Scales)] --> B
    A --> C
    A --> D

    B[256x256\nHigh Mag 1] --> E[EfficientNet Features]
    C[512x512\nHigh Mag 2] --> F[EfficientNet Features]
    D[1024x1024\nHigh Mag 3] --> G[EfficientNet Features]

    E --> H[Projection\nto Embed]
    F --> I[Projection\nto Embed]
    G --> J[Projection\nto Embed]

    H --> K[Cross-Attention\nFusion Layer]
    I --> K
    J --> K

    K --> L[Combined Features\n(Concatenated)]
    L --> M[Classifier\nMLP + Dropout]
    M --> N[Class\n0 or 1]

    style A fill:#f9f,stroke:#333
    style K fill:#ffdda8,stroke:#333
    style M fill:#d0e8ff,stroke:#333
    style N fill:#c8f4c8,stroke:#333
