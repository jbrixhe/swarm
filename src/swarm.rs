use crate::entity::Entity;
use crate::entity::EntityAllocator;
use crate::entity::EntityLocation;
use crate::entity::Locations;
use crate::filter::ArchetypeFilterData;
use crate::filter::ChunksetFilterData;
use crate::filter::Filter;
use crate::index::ArchetypeIndex;
use crate::index::ComponentIndex;
use crate::index::SetIndex;
use crate::storage::ArchetypeDescription;
use crate::storage::Chunk;
use crate::storage::Component;
use crate::storage::Storage;
use crate::storage::Tag;
use crate::storage::TagTypeId;
use crate::storage::Tags;
use crate::zip::Zip;
use std::iter::FusedIterator;
use std::iter::Peekable;
use std::ops::Deref;
use tracing::{span, trace, Level};

/// Contains queryable collections of data associated with `Entity`s.
pub struct Swarm<C: Component> {
    storage: Storage<C>,
    pub(crate) entity_allocator: EntityAllocator,
    entity_locations: Locations,
    pub(crate) allocation_buffer: Vec<Entity>,
}

unsafe impl<C: Component> Send for Swarm<C> {}

unsafe impl<C: Component> Sync for Swarm<C> {}

impl<C: Component> Swarm<C> {
    pub const DEFAULT_COMMAND_BUFFER_SIZE: usize = 64;

    /// Create a new `swarm` independent of any `Universe`.
    ///
    /// `Entity` IDs in such a swarm will only be unique within that swarm. See also
    /// `Universe::create_world`.r
    pub fn new() -> Self {
        Self {
            storage: Storage::new(),
            entity_allocator: EntityAllocator::new(),
            entity_locations: Locations::new(),
            allocation_buffer: Vec::with_capacity(Self::DEFAULT_COMMAND_BUFFER_SIZE),
        }
    }

    #[inline]
    pub fn push<T, CS>(&mut self, tags: T, components: CS) -> &[Entity]
    where
        T: TagSet + TagLayout + for<'a> Filter<ChunksetFilterData<'a, C>>,
        CS: IntoComponentSource<C>,
    {
        self.push_impl(tags, components.into())
    }

    pub(crate) fn push_impl<T, CS>(&mut self, mut tags: T, mut components: CS) -> &[Entity]
    where
        T: TagSet + TagLayout + for<'a> Filter<ChunksetFilterData<'a, C>>,
        CS: ComponentSource<C>,
    {
        let span = span!(Level::TRACE, "Inserting entities");
        let _guard = span.enter();

        // find or create archetype
        let archetype_index = self.find_or_create_archetype(&mut tags);

        // find or create chunk set
        let chunk_set_index = self.find_or_create_chunk(archetype_index, &mut tags);

        self.allocation_buffer.clear();
        self.allocation_buffer.reserve(components.len());

        // insert components into chunks
        while !components.is_empty() {
            // get chunk component storage
            let archetype = self.storage.archetype_mut(archetype_index).unwrap();
            let chunk_index = archetype.get_free_chunk(chunk_set_index, 1);
            let chunk = archetype
                .chunkset_mut(chunk_set_index)
                .unwrap()
                .chunk_mut(chunk_index)
                .unwrap();

            // insert as many components as we can into the chunk
            let allocated = components.write(self.entity_allocator.create_entities(), chunk);

            // record new entity locations
            let start = chunk.len() - allocated;
            let added = chunk.entities().iter().enumerate().skip(start);
            for (i, e) in added {
                let location = EntityLocation::new(
                    archetype_index,
                    chunk_set_index,
                    chunk_index,
                    ComponentIndex(i),
                );
                self.entity_locations.set(*e, location);
                self.allocation_buffer.push(*e);
            }
        }

        trace!(count = self.allocation_buffer.len(), "Inserted entities");

        &self.allocation_buffer
    }

    /// Removes the given `Entity` from the `swarm`.
    ///
    /// Returns `true` if the entity was deleted; else `false`.
    pub fn remove(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }

        if self.entity_allocator.delete_entity(entity) {
            let location = self.entity_locations.get(entity).unwrap();
            self.delete_location(location);
            trace!(?entity, "Deleted entity");
            true
        } else {
            false
        }
    }

    /// Delete all entity data. This leaves subscriptions and the command buffer intact.
    pub fn clear(&mut self) {
        for archetype in self.storage.archetypes_mut() {
            archetype.delete_all();
        }

        self.entity_allocator.delete_all_entities();
    }

    fn delete_location(&mut self, location: EntityLocation) {
        // find entity's chunk
        let chunk = self.storage.chunk_mut(location).unwrap();

        // swap remove with last entity in chunk
        if let Some(swapped) = chunk.swap_remove(location.component()) {
            // record swapped entity's new location
            self.entity_locations.set(swapped, location);
        }
    }

    fn get_component_storage(&self, entity: Entity) -> Option<&Chunk<C>> {
        let location = self.entity_locations.get(entity)?;
        self.storage.chunk(location)
    }

    fn find_archetype<T>(&self, tags: &mut T) -> Option<ArchetypeIndex>
    where
        T: for<'a> Filter<ArchetypeFilterData<'a>>,
    {
        // search for an archetype with an exact match for the desired component layout
        let archetype_data = ArchetypeFilterData {
            tag_types: self.storage.tag_types(),
        };

        // zip the two filters together - find the first index that matches both
        tags.matches(archetype_data)
            .enumerate()
            .take(self.storage.archetypes().len())
            .filter(|(_, a)| *a)
            .map(|(i, _)| i)
            .next()
            .map(ArchetypeIndex)
    }

    fn create_archetype<T>(&mut self, tags: &T) -> ArchetypeIndex
    where
        T: TagLayout,
    {
        let mut description = ArchetypeDescription::default();
        tags.tailor_archetype(&mut description);

        let (index, _) = self.storage.alloc_archetype(description);
        index
    }

    fn find_or_create_archetype<T>(&mut self, tags: &mut T) -> ArchetypeIndex
    where
        T: TagLayout,
    {
        if let Some(i) = self.find_archetype(tags.get_filter()) {
            i
        } else {
            self.create_archetype(tags)
        }
    }

    fn find_chunk_set<T>(&self, archetype: ArchetypeIndex, tags: &mut T) -> Option<SetIndex>
    where
        T: for<'a> Filter<ChunksetFilterData<'a, C>>,
    {
        // fetch the archetype, we can already assume that the archetype index is valid
        let archetype_data = self.storage.archetype(archetype).unwrap();

        // find a chunk with the correct tags
        let chunk_filter_data = ChunksetFilterData {
            archetype_data: archetype_data.deref(),
        };

        if let Some(i) = tags.matches(chunk_filter_data).matching_indices().next() {
            return Some(SetIndex(i));
        }

        None
    }

    fn create_chunk_set<T>(&mut self, archetype: ArchetypeIndex, tags: &T) -> SetIndex
    where
        T: TagSet,
    {
        let archetype_data = self.storage.archetype_mut(archetype).unwrap();
        archetype_data.alloc_chunk_set(|chunk_tags| tags.write_tags(chunk_tags))
    }

    fn find_or_create_chunk<T>(&mut self, archetype: ArchetypeIndex, tags: &mut T) -> SetIndex
    where
        T: TagSet + for<'a> Filter<ChunksetFilterData<'a, C>>,
    {
        if let Some(i) = self.find_chunk_set(archetype, tags) {
            i
        } else {
            self.create_chunk_set(archetype, tags)
        }
    }

    #[inline]
    fn get(&self, entity: Entity) -> Option<&C> {
        if !self.is_alive(entity) {
            return None;
        }

        let location = self.entity_locations.get(entity)?;
        let chunk = self.storage.chunk(location)?;
        chunk.get_component(location.component())
    }

    #[inline]
    fn is_alive(&self, entity: Entity) -> bool {
        self.entity_allocator.is_alive(entity)
    }
}

/// Describes the types of a set of tags attached to an entity.
pub trait TagLayout: Sized {
    /// A filter type which filters archetypes to an exact match with this layout.
    type Filter: for<'a> Filter<ArchetypeFilterData<'a>>;

    /// Gets the archetype filter for this layout.
    fn get_filter(&mut self) -> &mut Self::Filter;

    /// Modifies an archetype description to include the tags described by this layout.
    fn tailor_archetype(&self, archetype: &mut ArchetypeDescription);
}

/// A set of tag values to be attached to an entity.
pub trait TagSet {
    /// Writes the tags in this set to a new chunk.
    fn write_tags(&self, tags: &mut Tags);
}

/// A set of components to be attached to one or more entities.
pub trait ComponentSource<T: Component> {
    /// Determines if this component source has any more entity data to write.
    fn is_empty(&mut self) -> bool;

    /// Retrieves the nubmer of entities in this component source.
    fn len(&self) -> usize;

    /// Writes as many components as possible into a chunk.
    fn write<I: Iterator<Item = Entity>>(&mut self, entities: I, chunk: &mut Chunk<T>) -> usize;
}

/// An object that can be converted into a `ComponentSource`.
pub trait IntoComponentSource<T: Component> {
    /// The component source type that can be converted into.
    type Source: ComponentSource<T>;

    /// Converts `self` into a component source.
    fn into(self) -> Self::Source;
}

/// A `ComponentSource` which can insert tuples of components representing each entity into a swarm.
pub struct ComponentTupleSet<T: Component, I>
where
    I: Iterator<Item = T>,
{
    iter: Peekable<I>,
}

impl<T: Component, I> From<I> for ComponentTupleSet<T, I>
where
    I: Iterator<Item = T>,
    ComponentTupleSet<T, I>: ComponentSource<T>,
{
    fn from(iter: I) -> Self {
        ComponentTupleSet {
            iter: iter.peekable(),
        }
    }
}

impl<T, I> IntoComponentSource<T> for I
where
    T: Component,
    I: IntoIterator<Item = T>,
    ComponentTupleSet<I::Item, I::IntoIter>: ComponentSource<T>,
{
    type Source = ComponentTupleSet<I::Item, I::IntoIter>;

    fn into(self) -> Self::Source {
        ComponentTupleSet {
            iter: self.into_iter().peekable(),
        }
    }
}

impl<T, I> ComponentSource<T> for ComponentTupleSet<T, I>
where
    T: Component,
    I: ExactSizeIterator + Iterator<Item = T>,
{
    fn is_empty(&mut self) -> bool {
        self.iter.peek().is_none()
    }

    fn len(&self) -> usize {
        self.iter.len()
    }

    fn write<EntityIter: Iterator<Item = Entity>>(
        &mut self,
        mut allocator: EntityIter,
        chunk: &mut Chunk<T>,
    ) -> usize {
        let space = chunk.capacity() - chunk.len();
        let mut count = 0;

        while let Some(item) = {
            if count == space {
                None
            } else {
                self.iter.next()
            }
        } {
            let entity = allocator.next().unwrap();
            chunk.push(entity, item);
            count += 1;
        }

        count
    }
}

struct ConcatIter<'a, T, A: Iterator<Item = T> + FusedIterator, B: Iterator<Item = T>> {
    a: &'a mut A,
    b: &'a mut B,
}

impl<'a, T, A: Iterator<Item = T> + FusedIterator, B: Iterator<Item = T>> Iterator
    for ConcatIter<'a, T, A, B>
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.a.next().or_else(|| self.b.next())
    }
}

mod tuple_impls {
    use super::*;
    use crate::slice_vec::SliceVecIter;
    use crate::tuple::TupleEq;
    use std::iter::Repeat;
    use std::iter::Take;
    use std::slice::Iter;

    macro_rules! impl_data_tuple {
        ( $( $ty: ident => $id: ident ),* ) => {
            impl_data_tuple!(@TAG_SET $( $ty => $id ),*);
        };
        ( @TAG_SET $( $ty: ident => $id: ident ),* ) => {
            impl_data_tuple!(@CHUNK_FILTER $( $ty => $id ),*);

            impl<$( $ty ),*> TagSet for ($( $ty, )*)
            where
                $( $ty: Tag ),*
            {
                fn write_tags(&self, tags: &mut Tags) {
                    #![allow(unused_variables)]
                    #![allow(non_snake_case)]
                    let ($($id,)*) = self;
                    $(
                        unsafe {
                            tags.get_mut(TagTypeId::of::<$ty>())
                                .unwrap()
                                .push($id.clone())
                        };
                    )*
                }
            }

            impl <$( $ty ),*> TagLayout for ($( $ty, )*)
            where
                $( $ty: Tag ),*
            {
                type Filter = Self;

                fn get_filter(&mut self) -> &mut Self {
                    self
                }

                fn tailor_archetype(&self, archetype: &mut ArchetypeDescription) {
                    #![allow(unused_variables)]
                    $(
                        archetype.register_tag::<$ty>();
                    )*
                }
            }

            impl<'a, $( $ty ),*> Filter<ArchetypeFilterData<'a>> for ($( $ty, )*)
            where
                $( $ty: Tag ),*
            {
                type Iter = SliceVecIter<'a, TagTypeId>;

                fn collect(&self, source: ArchetypeFilterData<'a>) -> Self::Iter {
                    source.tag_types.iter()
                }

                fn is_match(&self, item: &<Self::Iter as Iterator>::Item) -> Option<bool> {
                    let types = &[$( TagTypeId::of::<$ty>() ),*];
                    Some(types.len() == item.len() && types.iter().all(|t| item.contains(t)))
                }
            }
        };
        ( @CHUNK_FILTER $( $ty: ident => $id: ident ),+ ) => {
            impl<'a, Comp: Component, $( $ty ),*> Filter<ChunksetFilterData<'a, Comp>> for ($( $ty, )*)
            where
                $( $ty: Tag ),*
            {
                type Iter = Zip<($( Iter<'a, $ty>, )*)>;

                fn collect(&self, source: ChunksetFilterData<'a, Comp>) -> Self::Iter {
                    let iters = (
                        $(
                            unsafe {
                                source.archetype_data
                                    .tags()
                                    .get(TagTypeId::of::<$ty>())
                                    .unwrap()
                                    .data_slice::<$ty>()
                                    .iter()
                            },
                        )*
                    );

                    crate::zip::multizip(iters)
                }

                fn is_match(&self, item: &<Self::Iter as Iterator>::Item) -> Option<bool> {
                    #![allow(non_snake_case)]
                    let ($( $ty, )*) = self;
                    Some(($( &*$ty, )*).legion_eq(item))
                }
            }
        };
        ( @CHUNK_FILTER ) => {
            impl<'a, Comp: Component> Filter<ChunksetFilterData<'a, Comp>> for () {
                type Iter = Take<Repeat<()>>;

                fn collect(&self, source: ChunksetFilterData<'a, Comp>) -> Self::Iter {
                    std::iter::repeat(()).take(source.archetype_data.len())
                }

                fn is_match(&self, _: &<Self::Iter as Iterator>::Item) -> Option<bool> {
                    Some(true)
                }
            }
        };
    }

    impl_data_tuple!();
    impl_data_tuple!(A => a);
    impl_data_tuple!(A => a, B => b);
    impl_data_tuple!(A => a, B => b, C => c);
    impl_data_tuple!(A => a, B => b, C => c, D => d);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m, N => n);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m, N => n, O => o);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m, N => n, O => o, P => p);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m, N => n, O => o, P => p, Q => q);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m, N => n, O => o, P => p, Q => q, R => r);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m, N => n, O => o, P => p, Q => q, R => r, S => s);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m, N => n, O => o, P => p, Q => q, R => r, S => s, T => t);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m, N => n, O => o, P => p, Q => q, R => r, S => s, T => t, U => u);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m, N => n, O => o, P => p, Q => q, R => r, S => s, T => t, U => u, V => v);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m, N => n, O => o, P => p, Q => q, R => r, S => s, T => t, U => u, V => v, W => w);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m, N => n, O => o, P => p, Q => q, R => r, S => s, T => t, U => u, V => v, W => w, X => x);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m, N => n, O => o, P => p, Q => q, R => r, S => s, T => t, U => u, V => v, W => w, X => x, Y => y);
    impl_data_tuple!(A => a, B => b, C => c, D => d, E => e, F => f, G => g, H => h, I => i, J => j, K => k, L => l, M => m, N => n, O => o, P => p, Q => q, R => r, S => s, T => t, U => u, V => v, W => w, X => x, Y => y, Z => z);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert() {
        #[derive(Debug, Eq, PartialEq)]
        struct One<'a> {
            msg: &'a str,
        };

        let mut swarm: Swarm<One> = Swarm::new();

        let entity = swarm.push((), vec![One { msg: "Bonjour" }])[0];

        assert_eq!(1, swarm.allocation_buffer.len());
        assert_eq!(*swarm.get(entity).unwrap(), One { msg: "Bonjour" });
    }
}
