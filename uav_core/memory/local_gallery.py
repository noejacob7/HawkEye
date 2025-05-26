"""
uav_core.memory.local_gallery

Stores captured images and their embeddings for a UAV.
"""

class LocalGallery:
    def __init__(self):
        # list of (image_id, embedding, metadata)
        self.entries = []

    def add_entry(self, image_id, embedding, metadata=None):
        """Add an image embedding to the gallery."""
        self.entries.append({'id': image_id, 'embedding': embedding, 'metadata': metadata})

    def get_all_embeddings(self):
        """Return a list of embeddings for all entries."""
        return [e['embedding'] for e in self.entries]

    def clear(self):
        """Clear the gallery entries."""
        self.entries.clear()
