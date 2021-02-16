use crate::entity::Entity;
use crate::entity::EntityLocation;
use crate::index::ArchetypeIndex;
use crate::index::ChunkIndex;
use crate::index::ComponentIndex;
use crate::index::SetIndex;
use crate::slice_vec::{SliceVec, SliceVecIter};
use derivative::Derivative;
use smallvec::SmallVec;
use std::any::TypeId;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::mem::size_of;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::RangeBounds;
use std::ptr::NonNull;
use std::slice::Iter;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use tracing::trace;

static VERSION_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_version() -> u64 {
    VERSION_COUNTER
        .fetch_add(1, Ordering::Relaxed)
        .checked_add(1)
        .unwrap()
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct TagTypeId(pub TypeId, pub u32);
impl TagTypeId {
    /// Gets the tag type ID that represents type `T`.
    pub fn of<T: Component>() -> Self {
        Self(TypeId::of::<T>(), 0)
    }

    pub fn type_id(&self) -> TypeId {
        self.0
    }
}

/// A `Component` is per-entity data that can be attached to a single entity.
pub trait Component: Send + Sync + 'static {}

/// A `Tag` is shared data that can be attached to multiple entities at once.
pub trait Tag: Clone + Send + Sync + PartialEq + 'static {}

impl<T: Send + Sync + 'static> Component for T {}
impl<T: Clone + Send + Sync + PartialEq + 'static> Tag for T {}

/// Stores slices of `TagTypeId`, each of which identifies the type of tags
/// contained within the archetype of the same index.
#[derive(Derivative)]
#[derivative(Default(bound = ""))]
pub struct TagTypes(SliceVec<TagTypeId>);

impl TagTypes {
    /// Gets an iterator over all type ID slices.
    pub fn iter(&self) -> SliceVecIter<TagTypeId> {
        self.0.iter()
    }

    /// Gets the number of slices stored within the set.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Determines if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.len() < 1
    }
}

/// Stores all entity data for a `World`.
pub struct Storage<T: Component> {
    tag_types: TagTypes,
    archetypes: Vec<ArchetypeData<T>>,
}

impl<T: Component> Storage<T> {
    // Creates an empty `Storage`.
    pub fn new() -> Self {
        Self {
            tag_types: TagTypes::default(),
            archetypes: Vec::default(),
        }
    }

    /// Creates a new archetype.
    ///
    /// Returns the index of the newly created archetype and an exclusive reference to the
    /// achetype's data.
    pub(crate) fn alloc_archetype(
        &mut self,
        desc: ArchetypeDescription,
    ) -> (ArchetypeIndex, &mut ArchetypeData<T>) {
        let index = ArchetypeIndex(self.archetypes.len());
        let id = ArchetypeId(index);
        let archetype = ArchetypeData::new(id, desc);

        self.push(archetype);

        let archetype = &mut self.archetypes[index];
        (index, archetype)
    }

    pub(crate) fn push(&mut self, archetype: ArchetypeData<T>) {
        let tags = archetype.tags();
        self.tag_types.0.push(tags.iter().map(|&(t, _)| t));

        let id = archetype.id();
        trace!(archetype = *id.index(), "Created Archetype");

        self.archetypes.push(archetype);
    }

    /// Gets a vector of slices of all tag types for all archetypes.
    ///
    /// Each slice contains the tag types for the archetype at the corresponding index.
    pub fn tag_types(&self) -> &TagTypes {
        &self.tag_types
    }

    /// Gets a slice reference to all archetypes.
    pub fn archetypes(&self) -> &[ArchetypeData<T>] {
        &self.archetypes
    }

    /// Gets a mutable slice reference to all archetypes.
    pub fn archetypes_mut(&mut self) -> &mut [ArchetypeData<T>] {
        &mut self.archetypes
    }

    pub(crate) fn drain<R: RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> std::vec::Drain<ArchetypeData<T>> {
        self.archetypes.drain(range)
    }

    pub(crate) fn archetype(
        &self,
        ArchetypeIndex(index): ArchetypeIndex,
    ) -> Option<&ArchetypeData<T>> {
        self.archetypes().get(index)
    }

    pub(crate) fn archetype_mut(
        &mut self,
        ArchetypeIndex(index): ArchetypeIndex,
    ) -> Option<&mut ArchetypeData<T>> {
        self.archetypes_mut().get_mut(index)
    }

    pub(crate) fn chunk(&self, loc: EntityLocation) -> Option<&Chunk<T>> {
        self.archetype(loc.archetype())
            .and_then(|atd| atd.chunkset(loc.set()))
            .and_then(|cs| cs.chunk(loc.chunk()))
    }

    pub(crate) fn chunk_mut(&mut self, loc: EntityLocation) -> Option<&mut Chunk<T>> {
        self.archetype_mut(loc.archetype())
            .and_then(|atd| atd.chunkset_mut(loc.set()))
            .and_then(|cs| cs.chunk_mut(loc.chunk()))
    }
}

/// Stores metadata decribing the type of a tag.
#[derive(Copy, Clone, PartialEq)]
pub struct TagMeta {
    size: usize,
    align: usize,
    drop_fn: Option<fn(*mut u8)>,
    eq_fn: fn(*const u8, *const u8) -> bool,
    clone_fn: fn(*const u8, *mut u8),
}

impl TagMeta {
    /// Gets the tag meta of tag type `T`.
    pub fn of<T: Tag>() -> Self {
        TagMeta {
            size: size_of::<T>(),
            align: std::mem::align_of::<T>(),
            drop_fn: if std::mem::needs_drop::<T>() {
                Some(|ptr| unsafe { std::ptr::drop_in_place(ptr as *mut T) })
            } else {
                None
            },
            eq_fn: |a, b| unsafe { *(a as *const T) == *(b as *const T) },
            clone_fn: |src, dst| unsafe {
                let clone = (&*(src as *const T)).clone();
                std::ptr::write(dst as *mut T, clone);
            },
        }
    }

    pub(crate) unsafe fn equals(&self, a: *const u8, b: *const u8) -> bool {
        (self.eq_fn)(a, b)
    }

    pub(crate) unsafe fn clone(&self, src: *const u8, dst: *mut u8) {
        (self.clone_fn)(src, dst)
    }

    pub(crate) unsafe fn drop(&self, val: *mut u8) {
        if let Some(drop_fn) = self.drop_fn {
            (drop_fn)(val);
        }
    }

    pub(crate) fn layout(&self) -> std::alloc::Layout {
        unsafe { std::alloc::Layout::from_size_align_unchecked(self.size, self.align) }
    }

    pub(crate) fn is_zero_sized(&self) -> bool {
        self.size == 0
    }
}

/// Describes the layout of an archetype, including what components
/// and tags shall be attached to entities stored within an archetype.
#[derive(Default, Clone, PartialEq)]
pub struct ArchetypeDescription {
    tags: Vec<(TagTypeId, TagMeta)>,
}

impl ArchetypeDescription {
    /// Gets a slice of the tags in the description.
    pub fn tags(&self) -> &[(TagTypeId, TagMeta)] {
        &self.tags
    }

    /// Adds a tag to the description.
    pub fn register_tag_raw(&mut self, type_id: TagTypeId, type_meta: TagMeta) {
        self.tags.push((type_id, type_meta));
    }

    /// Adds a tag to the description.
    pub fn register_tag<T: Tag>(&mut self) {
        self.tags.push((TagTypeId::of::<T>(), TagMeta::of::<T>()));
    }
}

/// Unique ID of an archetype.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ArchetypeId(ArchetypeIndex);

impl ArchetypeId {
    pub(crate) fn new(index: ArchetypeIndex) -> Self {
        ArchetypeId(index)
    }

    pub fn index(self) -> ArchetypeIndex {
        self.0
    }
}

/// Contains all of the tags attached to the entities in each chunk.
pub struct Tags(pub(crate) SmallVec<[(TagTypeId, TagStorage); 3]>);

impl Tags {
    fn new(mut data: SmallVec<[(TagTypeId, TagStorage); 3]>) -> Self {
        data.sort_by_key(|&(t, _)| t);
        Self(data)
    }

    fn iter(&self) -> Iter<(TagTypeId, TagStorage)> {
        self.0.iter()
    }

    fn validate(&self, set_count: usize) {
        for (_, tags) in self.0.iter() {
            debug_assert_eq!(set_count, tags.len());
        }
    }

    /// Gets the set of tag values of the specified type attached to all chunks.
    #[inline]
    pub fn get(&self, type_id: TagTypeId) -> Option<&TagStorage> {
        self.0
            .binary_search_by_key(&type_id, |&(t, _)| t)
            .ok()
            .map(|i| unsafe { &self.0.get_unchecked(i).1 })
    }

    /// Mutably gets the set of all tag values of the specified type attached to all chunks.
    #[inline]
    pub fn get_mut(&mut self, type_id: TagTypeId) -> Option<&mut TagStorage> {
        self.0
            .binary_search_by_key(&type_id, |&(t, _)| t)
            .ok()
            .map(move |i| unsafe { &mut self.0.get_unchecked_mut(i).1 })
    }
}

/// Stores entity data in chunks. All entities within an archetype have the same data layout
/// (component and tag types).
pub struct ArchetypeData<T: Component> {
    id: ArchetypeId,
    tags: Tags,
    chunk_sets: Vec<Chunkset<T>>,
}

impl<T: Component> ArchetypeData<T> {
    fn new(id: ArchetypeId, desc: ArchetypeDescription) -> Self {
        // create tag storage
        let tags = desc
            .tags
            .iter()
            .map(|&(type_id, meta)| (type_id, TagStorage::new(meta)))
            .collect();

        ArchetypeData {
            id,
            tags: Tags::new(tags),
            chunk_sets: Vec::new(),
        }
    }

    pub(crate) fn delete_all(&mut self) {
        for set in &mut self.chunk_sets {
            // Clearing the chunk will Drop all the data
            set.chunks.clear();
        }
    }

    /// Gets the unique ID of this archetype.
    pub fn id(&self) -> ArchetypeId {
        self.id
    }

    fn find_chunk_set_by_tags(
        &self,
        other_tags: &Tags,
        other_set_index: SetIndex,
    ) -> Option<SetIndex> {
        // search for a matching chunk set
        let mut set_match = None;
        for self_set_index in 0..self.chunk_sets.len() {
            let self_set_index = SetIndex(self_set_index);
            let mut matches = true;
            for &(type_id, ref tags) in self.tags.0.iter() {
                unsafe {
                    let (self_tag_ptr, size, _) = tags.data_raw();
                    let (other_tag_ptr, _, _) = other_tags.get(type_id).unwrap().data_raw();

                    if !tags.element().equals(
                        self_tag_ptr.as_ptr().add(self_set_index.0 * size),
                        other_tag_ptr.as_ptr().add(other_set_index.0 * size),
                    ) {
                        matches = false;
                        break;
                    }
                }
            }

            if matches {
                set_match = Some(self_set_index);
                break;
            }
        }

        set_match
    }

    pub(crate) fn find_or_create_chunk_set_by_tags(
        &mut self,
        src_tags: &Tags,
        src_chunk_set_index: SetIndex,
    ) -> SetIndex {
        let dst_chunk_set_index = self.find_chunk_set_by_tags(src_tags, src_chunk_set_index);
        dst_chunk_set_index.unwrap_or_else(|| {
            self.alloc_chunk_set(|self_tags| {
                for (type_id, other_tags) in src_tags.0.iter() {
                    unsafe {
                        let (src, _, _) = other_tags.data_raw();
                        let dst = self_tags.get_mut(*type_id).unwrap().alloc_ptr();
                        other_tags.element().clone(src.as_ptr(), dst);
                    }
                }
            })
        })
    }

    pub(crate) fn move_from(&mut self, mut other: ArchetypeData<T>) {
        let other_tags = &other.tags;
        for (other_index, mut set) in other.chunk_sets.drain(..).enumerate() {
            let other_index = SetIndex(other_index);
            let set_match = self.find_chunk_set_by_tags(&other_tags, other_index);

            if let Some(chunk_set) = set_match {
                // if we found a match, move the chunks into the set
                let target = &mut self.chunk_sets[chunk_set];
                for chunk in set.drain(..) {
                    target.push(chunk);
                }
            } else {
                // if we did not find a match, clone the tags and move the set
                self.push(set, |self_tags| {
                    for &(type_id, ref other_tags) in other_tags.0.iter() {
                        unsafe {
                            let (src, _, _) = other_tags.data_raw();
                            let dst = self_tags.get_mut(type_id).unwrap().alloc_ptr();
                            other_tags.element().clone(src.as_ptr(), dst);
                        }
                    }
                });
            }
        }

        self.tags.validate(self.chunk_sets.len());
    }

    /// Iterate all entities in existence by iterating across archetypes, chunk sets, and chunks
    pub(crate) fn iter_entities<'a>(&'a self) -> impl Iterator<Item = Entity> + 'a {
        self.chunk_sets.iter().flat_map(move |set| {
            set.chunks
                .iter()
                .flat_map(move |chunk| chunk.entities().iter().copied())
        })
    }

    pub(crate) fn iter_entity_locations<'a>(
        &'a self,
        archetype_index: ArchetypeIndex,
    ) -> impl Iterator<Item = (Entity, EntityLocation)> + 'a {
        self.chunk_sets
            .iter()
            .enumerate()
            .flat_map(move |(set_index, set)| {
                set.chunks
                    .iter()
                    .enumerate()
                    .flat_map(move |(chunk_index, chunk)| {
                        chunk
                            .entities()
                            .iter()
                            .enumerate()
                            .map(move |(entity_index, &entity)| {
                                (
                                    entity,
                                    EntityLocation::new(
                                        archetype_index,
                                        SetIndex(set_index),
                                        ChunkIndex(chunk_index),
                                        ComponentIndex(entity_index),
                                    ),
                                )
                            })
                    })
            })
    }

    fn push<F: FnMut(&mut Tags)>(&mut self, set: Chunkset<T>, mut initialize: F) {
        initialize(&mut self.tags);
        self.chunk_sets.push(set);

        self.tags.validate(self.chunk_sets.len());
    }

    /// Allocates a new chunk set. Returns the index of the new set.
    ///
    /// `initialize` is expected to push the new chunkset's tag values onto the tags collection.
    pub(crate) fn alloc_chunk_set<F: FnMut(&mut Tags)>(&mut self, initialize: F) -> SetIndex {
        self.push(Chunkset::new(), initialize);
        SetIndex(self.chunk_sets.len() - 1)
    }

    /// Finds a chunk with space free for at least `minimum_space` entities, creating a chunk if needed.
    pub(crate) fn get_free_chunk(
        &mut self,
        set_index: SetIndex,
        minimum_space: usize,
    ) -> ChunkIndex {
        let chunk_index = {
            let chunks = &mut self.chunk_sets[set_index];
            let len = chunks.len();
            for (i, chunk) in chunks.iter_mut().enumerate() {
                let space_left = chunk.capacity() - chunk.len();
                if space_left >= minimum_space {
                    return ChunkIndex(i);
                }
            }
            ChunkIndex(len)
        };

        let component_storage = Chunk::new(ChunkId(self.id, set_index, chunk_index));
        self.chunkset_mut(set_index)
            .unwrap()
            .push(component_storage);

        trace!(
            archetype = *self.id.index(),
            chunkset = *set_index,
            chunk = *chunk_index,
            "Created chunk"
        );

        chunk_index
    }

    /// Gets the number of chunk sets stored within this archetype.
    pub fn len(&self) -> usize {
        self.chunk_sets.len()
    }

    /// Determines whether this archetype has any chunks.
    pub fn is_empty(&self) -> bool {
        self.len() < 1
    }

    /// Gets the tag storage for all chunks in the archetype.
    pub fn tags(&self) -> &Tags {
        &self.tags
    }

    /// Mutably gets the tag storage for all chunks in the archetype.
    pub fn tags_mut(&mut self) -> &mut Tags {
        &mut self.tags
    }

    /// Gets a slice of chunksets.
    pub fn chunksets(&self) -> &[Chunkset<T>] {
        &self.chunk_sets
    }

    /// Gets a mutable slice of chunksets.
    pub fn chunksets_mut(&mut self) -> &mut [Chunkset<T>] {
        &mut self.chunk_sets
    }

    pub(crate) fn chunkset(&self, SetIndex(index): SetIndex) -> Option<&Chunkset<T>> {
        self.chunksets().get(index)
    }

    pub(crate) fn chunkset_mut(&mut self, SetIndex(index): SetIndex) -> Option<&mut Chunkset<T>> {
        self.chunksets_mut().get_mut(index)
    }
}

/// Contains chunks with the same layout and tag values.
#[derive(Default)]
pub struct Chunkset<T: Component> {
    chunks: Vec<Chunk<T>>,
}

impl<T: Component> Deref for Chunkset<T> {
    type Target = [Chunk<T>];

    fn deref(&self) -> &Self::Target {
        self.chunks.as_slice()
    }
}

impl<T: Component> DerefMut for Chunkset<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.chunks.as_mut_slice()
    }
}

impl<T: Component> Chunkset<T> {
    pub(crate) fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    /// Pushes a new chunk into the set.
    pub fn push(&mut self, chunk: Chunk<T>) {
        self.chunks.push(chunk);
    }

    pub(crate) fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> std::vec::Drain<Chunk<T>> {
        self.chunks.drain(range)
    }

    /// Gets a slice reference to occupied chunks.
    pub fn occupied(&self) -> &[Chunk<T>] {
        let mut len = self.chunks.len();
        while len > 0 {
            if unsafe { !self.chunks.get_unchecked(len - 1).is_empty() } {
                break;
            }
            len -= 1;
        }
        let (some, _) = self.chunks.as_slice().split_at(len);
        some
    }

    /// Gets a mutable slice reference to occupied chunks.
    pub fn occupied_mut(&mut self) -> &mut [Chunk<T>] {
        let mut len = self.chunks.len();
        while len > 0 {
            if unsafe { !self.chunks.get_unchecked(len - 1).is_empty() } {
                break;
            }
            len -= 1;
        }
        let (some, _) = self.chunks.as_mut_slice().split_at_mut(len);
        some
    }

    pub(crate) fn chunk(&self, ChunkIndex(index): ChunkIndex) -> Option<&Chunk<T>> {
        self.chunks.get(index)
    }

    pub(crate) fn chunk_mut(&mut self, ChunkIndex(index): ChunkIndex) -> Option<&mut Chunk<T>> {
        self.chunks.get_mut(index)
    }

    pub(crate) unsafe fn chunk_unchecked(&self, ChunkIndex(index): ChunkIndex) -> &Chunk<T> {
        self.chunks.get_unchecked(index)
    }

    pub(crate) unsafe fn chunk_unchecked_mut(
        &mut self,
        ChunkIndex(index): ChunkIndex,
    ) -> &mut Chunk<T> {
        self.chunks.get_unchecked_mut(index)
    }
}

/// Unique ID of a chunk.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ChunkId(ArchetypeId, SetIndex, ChunkIndex);

impl ChunkId {
    pub(crate) fn new(archetype: ArchetypeId, set: SetIndex, index: ChunkIndex) -> Self {
        ChunkId(archetype, set, index)
    }

    pub fn archetype_id(&self) -> ArchetypeId {
        self.0
    }

    pub(crate) fn set(&self) -> SetIndex {
        self.1
    }

    pub(crate) fn index(&self) -> ChunkIndex {
        self.2
    }
}

/// Stores a chunk of entities and their component data of a specific data layout.
pub struct Chunk<T: Component> {
    id: ChunkId,
    capacity: usize,
    entities: Vec<Entity>,
    component_info: Vec<T>,
}

impl<T: Component> Chunk<T> {
    fn new(id: ChunkId) -> Self {
        Self {
            id,
            capacity: 1024,
            entities: Vec::with_capacity(1024),
            component_info: Vec::with_capacity(1024),
        }
    }
    /// Gets the unique ID of the chunk.
    pub fn id(&self) -> ChunkId {
        self.id
    }

    /// Gets the number of entities stored in the chunk.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Gets the maximum number of entities that can be stored in the chunk.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Determines if the chunk is full.
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Determines if the chunk is empty.
    pub fn is_empty(&self) -> bool {
        self.entities.len() == 0
    }

    /// Gets a slice reference containing the IDs of all entities stored in the chunk.
    pub fn entities(&self) -> &[Entity] {
        self.entities.as_slice()
    }

    /// Gets a component accessor for the specified component type.
    pub fn get_component(&self, ComponentIndex(index): ComponentIndex) -> Option<&T> {
        self.component_info.get(index)
    }

    /// Removes an entity from the chunk by swapping it with the last entry.
    ///
    /// Returns the ID of the entity which was swapped into the removed entity's position.
    pub fn swap_remove(&mut self, ComponentIndex(index): ComponentIndex) -> Option<Entity> {
        let _ = self.entities.swap_remove(index);
        let _ = self.component_info.swap_remove(index);
        if self.entities.len() > index {
            Some(*self.entities.get(index).unwrap())
        } else {
            if self.is_empty() {
                self.free();
            }

            None
        }
    }

    /// Gets mutable references to the internal data of the chunk.
    pub fn push(&mut self, entity: Entity, component: T) {
        self.entities.push(entity);
        self.component_info.push(component);
    }

    fn free(&mut self) {
        debug_assert_eq!(0, self.len());

        self.entities.shrink_to_fit();
        self.component_info.shrink_to_fit();

        trace!(
            archetype = *self.id.archetype_id().index(),
            chunkset = *self.id.set(),
            chunk = *self.id.index(),
            "Freeing chunk memory"
        );
    }
}

unsafe impl<T: Component> Sync for Chunk<T> {}

unsafe impl<T: Component> Send for Chunk<T> {}

/// A vector of tag values of a single type.
///
/// Each element in the vector represents the value of tag for
/// the chunk with the corresponding index.
pub struct TagStorage {
    ptr: NonNull<u8>,
    capacity: usize,
    len: usize,
    element: TagMeta,
}

impl TagStorage {
    pub(crate) fn new(element: TagMeta) -> Self {
        let capacity = if element.size == 0 { !0 } else { 4 };

        let ptr = unsafe {
            if element.size > 0 {
                let layout =
                    std::alloc::Layout::from_size_align(capacity * element.size, element.align)
                        .unwrap();
                NonNull::new_unchecked(std::alloc::alloc(layout))
            } else {
                NonNull::new_unchecked(element.align as *mut u8)
            }
        };

        TagStorage {
            ptr,
            capacity,
            len: 0,
            element,
        }
    }

    /// Gets the element metadata.
    pub fn element(&self) -> &TagMeta {
        &self.element
    }

    /// Gets the number of tags contained within the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Determines if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len() < 1
    }

    /// Allocates uninitialized memory for a new element.
    ///
    /// # Safety
    ///
    /// A valid element must be written into the returned address before the
    /// tag storage is next accessed.
    pub unsafe fn alloc_ptr(&mut self) -> *mut u8 {
        if self.len == self.capacity {
            self.grow();
        }

        let ptr = if self.element.size > 0 {
            self.ptr.as_ptr().add(self.len * self.element.size)
        } else {
            self.element.align as *mut u8
        };

        self.len += 1;
        ptr
    }

    /// Pushes a new tag onto the end of the vector.
    ///
    /// # Safety
    ///
    /// Ensure the tag pointed to by `ptr` is representative
    /// of the tag types stored in the vec.
    ///
    /// `ptr` must not point to a location already within the vector.
    ///
    /// The value at `ptr` is _copied_ into the tag vector. If the value
    /// is not `Copy`, then the caller must ensure that the original value
    /// is forgotten with `mem::forget` such that the finalizer is not called
    /// twice.
    pub unsafe fn push_raw(&mut self, ptr: *const u8) {
        if self.len == self.capacity {
            self.grow();
        }

        if self.element.size > 0 {
            let dst = self.ptr.as_ptr().add(self.len * self.element.size);
            std::ptr::copy_nonoverlapping(ptr, dst, self.element.size);
        }

        self.len += 1;
    }

    /// Pushes a new tag onto the end of the vector.
    ///
    /// # Safety
    ///
    /// Ensure that the type `T` is representative of the tag type stored in the vec.
    pub unsafe fn push<T: Tag>(&mut self, value: T) {
        debug_assert!(
            size_of::<T>() == self.element.size,
            "incompatible element data size"
        );
        self.push_raw(&value as *const T as *const u8);
        std::mem::forget(value);
    }

    /// Gets a raw pointer to the start of the tag slice.
    ///
    /// Returns a tuple containing `(pointer, element_size, count)`.
    ///
    /// # Safety
    /// This function returns a raw pointer with the size and length.
    /// Ensure that you do not access outside these bounds for this pointer.
    pub unsafe fn data_raw(&self) -> (NonNull<u8>, usize, usize) {
        (self.ptr, self.element.size, self.len)
    }

    /// Gets a shared reference to the slice of tags.
    ///
    /// # Safety
    ///
    /// Ensure that `T` is representative of the tag data actually stored.
    ///
    /// Access to the tag data within the slice is runtime borrow checked.
    /// This call will panic if borrowing rules are broken.
    pub unsafe fn data_slice<T>(&self) -> &[T] {
        debug_assert!(
            size_of::<T>() == self.element.size,
            "incompatible element data size"
        );
        std::slice::from_raw_parts(self.ptr.as_ptr() as *const T, self.len)
    }

    /// Drop the storage without dropping the tags contained in the storage
    pub(crate) fn forget_data(mut self) {
        // this is a bit of a hack, but it makes the Drop impl not drop the elements
        self.element.drop_fn = None;
    }

    fn grow(&mut self) {
        assert!(self.element.size != 0, "capacity overflow");
        unsafe {
            let (new_cap, ptr) = {
                let layout = std::alloc::Layout::from_size_align(
                    self.capacity * self.element.size,
                    self.element.align,
                )
                .unwrap();
                let new_cap = 2 * self.capacity;
                let ptr =
                    std::alloc::realloc(self.ptr.as_ptr(), layout, new_cap * self.element.size);

                (new_cap, ptr)
            };

            if ptr.is_null() {
                tracing::error!("out of memory");
                std::process::abort()
            }

            self.ptr = NonNull::new_unchecked(ptr);
            self.capacity = new_cap;
        }
    }
}

unsafe impl Sync for TagStorage {}

unsafe impl Send for TagStorage {}

impl Drop for TagStorage {
    fn drop(&mut self) {
        if self.element.size > 0 {
            let ptr = self.ptr.as_ptr();

            unsafe {
                if let Some(drop_fn) = self.element.drop_fn {
                    for i in 0..self.len {
                        drop_fn(ptr.add(i * self.element.size));
                    }
                }
                let layout = std::alloc::Layout::from_size_align_unchecked(
                    self.element.size * self.capacity,
                    self.element.align,
                );
                std::alloc::dealloc(ptr, layout);
            }
        }
    }
}

impl Debug for TagStorage {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "TagStorage {{ element_size: {}, count: {}, capacity: {} }}",
            self.element.size, self.len, self.capacity
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Copy, Clone, PartialEq, Debug)]
    struct ZeroSize;
}
