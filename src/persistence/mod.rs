use crate::{error::MemoryError, types::{MemoryId, MemoryItem}};
use parking_lot::RwLock;
use std::{collections::HashMap, sync::Arc};

pub trait MemoryStore: Send + Sync {
    fn save_sync(&self, item: &MemoryItem) -> Result<(), MemoryError>;
    fn load_sync(&self, id: &MemoryId) -> Result<MemoryItem, MemoryError>;
    fn all_sync(&self) -> Result<Vec<MemoryItem>, MemoryError>;
    fn delete_sync(&self, id: &MemoryId) -> Result<(), MemoryError>;
}

pub struct InMemoryStore {
    items: Arc<RwLock<HashMap<String, MemoryItem>>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self { items: Arc::new(RwLock::new(HashMap::new())) }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStore for InMemoryStore {
    fn save_sync(&self, item: &MemoryItem) -> Result<(), MemoryError> {
        self.items.write().insert(item.id.to_string(), item.clone());
        Ok(())
    }

    fn load_sync(&self, id: &MemoryId) -> Result<MemoryItem, MemoryError> {
        self.items
            .read()
            .get(&id.to_string())
            .cloned()
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))
    }

    fn all_sync(&self) -> Result<Vec<MemoryItem>, MemoryError> {
        Ok(self.items.read().values().cloned().collect())
    }

    fn delete_sync(&self, id: &MemoryId) -> Result<(), MemoryError> {
        self.items
            .write()
            .remove(&id.to_string())
            .map(|_| ())
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))
    }
}

pub struct FileStore {
    path: std::path::PathBuf,
    items: Arc<RwLock<HashMap<String, MemoryItem>>>,
}

impl FileStore {
    pub fn new(path: std::path::PathBuf) -> Result<Self, MemoryError> {
        let items = if path.exists() {
            let data = std::fs::read_to_string(&path)?;
            serde_json::from_str(&data)
                .map_err(|e| MemoryError::Serialization(e.to_string()))?
        } else {
            HashMap::new()
        };
        Ok(Self { path, items: Arc::new(RwLock::new(items)) })
    }

    pub fn flush(&self) -> Result<(), MemoryError> {
        let data = serde_json::to_string(&*self.items.read())
            .map_err(|e| MemoryError::Serialization(e.to_string()))?;
        std::fs::write(&self.path, data)?;
        Ok(())
    }
}

impl MemoryStore for FileStore {
    fn save_sync(&self, item: &MemoryItem) -> Result<(), MemoryError> {
        self.items.write().insert(item.id.to_string(), item.clone());
        Ok(())
    }

    fn load_sync(&self, id: &MemoryId) -> Result<MemoryItem, MemoryError> {
        self.items
            .read()
            .get(&id.to_string())
            .cloned()
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))
    }

    fn all_sync(&self) -> Result<Vec<MemoryItem>, MemoryError> {
        Ok(self.items.read().values().cloned().collect())
    }

    fn delete_sync(&self, id: &MemoryId) -> Result<(), MemoryError> {
        self.items
            .write()
            .remove(&id.to_string())
            .map(|_| ())
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentId;
    use tempfile::NamedTempFile;

    fn make_item() -> MemoryItem {
        MemoryItem::new(serde_json::json!("hello"), vec!["t".into()], AgentId::new("a"), 0.7)
    }

    #[test]
    fn test_in_memory_save_and_load() {
        let store = InMemoryStore::new();
        let item = make_item();
        let id = item.id.clone();
        store.save_sync(&item).expect("save");
        let loaded = store.load_sync(&id).expect("load");
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.tags, item.tags);
    }

    #[test]
    fn test_in_memory_load_missing_returns_not_found() {
        let store = InMemoryStore::new();
        let result = store.load_sync(&MemoryId::new());
        assert!(matches!(result, Err(MemoryError::NotFound(_))));
    }

    #[test]
    fn test_in_memory_delete() {
        let store = InMemoryStore::new();
        let item = make_item();
        let id = item.id.clone();
        store.save_sync(&item).expect("save");
        store.delete_sync(&id).expect("delete");
        assert!(matches!(store.load_sync(&id), Err(MemoryError::NotFound(_))));
    }

    #[test]
    fn test_in_memory_all() {
        let store = InMemoryStore::new();
        store.save_sync(&make_item()).expect("save 1");
        store.save_sync(&make_item()).expect("save 2");
        assert_eq!(store.all_sync().expect("all").len(), 2);
    }

    #[test]
    fn test_file_store_save_flush_and_reload() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("memory.json");

        let item = make_item();
        let id = item.id.clone();

        {
            let store = FileStore::new(path.clone()).expect("new");
            store.save_sync(&item).expect("save");
            store.flush().expect("flush");
        }

        // Load from fresh instance
        let store2 = FileStore::new(path).expect("reload");
        let loaded = store2.load_sync(&id).expect("load");
        assert_eq!(loaded.id, id);
    }

    #[test]
    fn test_file_store_empty_path_starts_fresh() {
        let tmp = NamedTempFile::new().expect("tempfile");
        // Remove file so store starts from scratch
        std::fs::remove_file(tmp.path()).ok();
        let store = FileStore::new(tmp.path().to_path_buf()).expect("new from empty");
        assert!(store.all_sync().expect("all").is_empty());
    }
}
