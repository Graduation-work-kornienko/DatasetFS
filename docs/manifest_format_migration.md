# DatasetFS Manifest Format Migration Guide

This document outlines the migration from JSON to Parquet format for DatasetFS manifests, including the rationale, implementation details, and usage instructions.

## 1. Rationale for Migration

The migration from JSON to Parquet format for DatasetFS manifests is driven by several key factors:

- **Storage Efficiency**: Parquet's columnar storage and compression provide significantly smaller file sizes compared to JSON
- **Performance**: Faster read/write operations due to binary format and efficient encoding
- **Remote Storage Optimization**: Parquet is designed for cloud storage systems like S3 with excellent integration
- **Predicate Pushdown**: Ability to filter data during read operations, reducing data transfer
- **Schema Evolution**: Support for schema changes over time

## 2. Format Details

### JSON Format (Legacy)

- File name: `metadata.jsonl`
- Structure: JSON object with version, shards_meta, and files fields
- Human-readable but verbose
- No built-in compression

### Parquet Format (Current)

- File name: `metadata.parquet`
- Structure: Columnar storage with efficient encoding
- Automatic compression (typically 5-10x smaller than JSON)
- Support for predicate pushdown and columnar operations

## 3. Implementation Details

The migration implements dual-format support with backward compatibility:

1. **Priority Reading**: The system first attempts to read `metadata.parquet`
2. **Fallback**: If Parquet file doesn't exist, falls back to `metadata.jsonl`
3. **Priority Writing**: The system attempts to write to `metadata.parquet` first
4. **Fallback**: If Parquet write fails, falls back to `metadata.jsonl`

This approach ensures smooth transition and backward compatibility.

## 4. Migration Utility

A command-line utility is provided to convert existing JSON manifests to Parquet format:

```bash
# Convert a JSON manifest to Parquet format
datasetconverter convert-manifest --source /path/to/dataset
```

The utility:
- Reads the JSON manifest from `metadata.jsonl`
- Converts the data to Parquet format
- Writes the result to `metadata.parquet`
- Preserves the original JSON file for backup

## 5. Usage Instructions

### For New Datasets

New datasets will automatically use Parquet format for manifests. No special action is required.

### For Existing Datasets

To migrate an existing dataset from JSON to Parquet format:

1. Run the conversion command:
```bash
$ datasetconverter convert-manifest --source /path/to/existing/dataset
```

2. Verify the conversion was successful:
```bash
$ ls -la /path/to/existing/dataset/metadata*
-rw-r--r--  1 user  group  500M metadata.jsonl
-rw-r--r--  1 user  group   50M metadata.parquet
```

3. The system will now use the Parquet format automatically.

## 6. Remote Storage Integration

The Parquet format is particularly beneficial for remote storage scenarios:

- **Reduced Bandwidth**: Smaller file sizes mean less data transfer
- **Efficient Access**: Predicate pushdown allows filtering data during read operations
- **Cloud-Native**: Optimized for distributed storage systems
- **Random Access**: Can read specific columns or row groups without loading entire file

When using remote storage (S3, GCS, etc.), the manifest should be stored in Parquet format to maximize these benefits.

## 7. Future Considerations

### WAL Format

While the manifest has been migrated to Parquet, the WAL (Write-Ahead Log) still uses JSONL format. Future work will address this by:

- Implementing a binary format for WAL to improve performance
- Maintaining durability guarantees
- Ensuring crash recovery capabilities

### Schema Evolution

The Parquet format supports schema evolution, allowing future additions to the manifest structure without breaking compatibility. Any schema changes should follow these guidelines:

- Add new fields as optional
- Maintain backward compatibility
- Document changes in this guide

## 8. Troubleshooting

### Conversion Issues

If the conversion utility fails:

1. Verify the source directory exists and contains a valid `metadata.jsonl` file
2. Check file permissions
3. Ensure sufficient disk space
4. Verify Parquet library is properly installed

### Compatibility Issues

If you encounter issues with older clients:

1. Ensure the JSON manifest (`metadata.jsonl`) is still present
2. The system will automatically fall back to JSON format if Parquet is not available
3. Consider running the conversion utility on the dataset

## 9. Conclusion

The migration to Parquet format significantly improves DatasetFS's storage efficiency and performance, particularly in remote storage environments. The dual-format support ensures backward compatibility while enabling the benefits of modern columnar storage.
