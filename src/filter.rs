use crate::storage::ArchetypeData;
use crate::storage::Chunk;
use crate::storage::Component;
// use crate::storage::ComponentTypes;
use crate::storage::TagTypes;
use std::marker::PhantomData;

/// A streaming iterator of bools.
pub trait Filter<T>: Send + Sync + Sized {
    type Iter: Iterator + Send + Sync;

    // Called when a query is about to begin execution.
    fn init(&self) {}

    /// Pulls iterator data out of the source.
    fn collect(&self, source: T) -> Self::Iter;

    /// Determines if an element of `Self::Iter` matches the filter conditions.
    fn is_match(&self, item: &<Self::Iter as Iterator>::Item) -> Option<bool>;

    /// Creates an iterator which yields bools for each element in the source
    /// which indicate if the element matches the filter.
    fn matches(&mut self, source: T) -> FilterIter<Self, T> {
        FilterIter {
            elements: self.collect(source),
            filter: self,
            _phantom: PhantomData,
        }
    }
}

/// An iterator over the elements matching a filter.
pub struct FilterIter<'a, F: Filter<T>, T> {
    elements: <F as Filter<T>>::Iter,
    filter: &'a mut F,
    _phantom: PhantomData<T>,
}

impl<'a, F: Filter<T>, T> Iterator for FilterIter<'a, F, T> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        self.elements
            .next()
            .map(|x| self.filter.is_match(&x).unwrap_or(false))
    }
}

impl<'a, F: Filter<T>, T: 'a> FilterIter<'a, F, T> {
    /// Finds the indices of all elements matching the filter.
    pub fn matching_indices(self) -> impl Iterator<Item = usize> + 'a {
        self.enumerate().filter(|(_, x)| *x).map(|(i, _)| i)
    }
}

/// Input data for archetype filters.
#[derive(Clone)]
pub struct ArchetypeFilterData<'a> {
    /// The tag types in each archetype.
    pub tag_types: &'a TagTypes,
}

/// Input data for chunkset filters.
#[derive(Clone)]
pub struct ChunksetFilterData<'a, T: Component> {
    /// The component data in an archetype.
    pub archetype_data: &'a ArchetypeData<T>,
}

/// Input data for chunk filters.
#[derive(Clone)]
pub struct ChunkFilterData<'a, T: Component> {
    // The components in a set
    pub chunks: &'a [Chunk<T>],
}

#[cfg(test)]
mod test {}
