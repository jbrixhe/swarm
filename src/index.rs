use crate::entity::Entity;
use crate::storage::ArchetypeData;
use crate::storage::Chunk;
use crate::storage::Chunkset;
use crate::storage::Component;
use std::fmt;
use std::ops::Deref;
use std::ops::Index;
use std::ops::IndexMut;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct SetIndex(pub usize);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ChunkIndex(pub usize);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ArchetypeIndex(pub usize);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ComponentIndex(pub usize);

macro_rules! impl_index {
    ($index_ty:ty: $output_ty:ty) => {
        impl<T: Component> Index<$index_ty> for [$output_ty] {
            type Output = $output_ty;
            #[inline(always)]
            fn index(&self, index: $index_ty) -> &Self::Output {
                &self[index.0]
            }
        }
        impl<T: Component> IndexMut<$index_ty> for [$output_ty] {
            #[inline(always)]
            fn index_mut(&mut self, index: $index_ty) -> &mut Self::Output {
                &mut self[index.0]
            }
        }
        impl<T: Component> Index<$index_ty> for Vec<$output_ty> {
            type Output = $output_ty;
            #[inline(always)]
            fn index(&self, index: $index_ty) -> &Self::Output {
                &self[index.0]
            }
        }
        impl<T: Component> IndexMut<$index_ty> for Vec<$output_ty> {
            #[inline(always)]
            fn index_mut(&mut self, index: $index_ty) -> &mut Self::Output {
                &mut self[index.0]
            }
        }
        impl Deref for $index_ty {
            type Target = usize;
            #[inline(always)]
            fn deref(&self) -> &usize {
                &self.0
            }
        }
        impl fmt::Display for $index_ty {
            #[inline]
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&**self, f)
            }
        }
    };
}

impl_index!(SetIndex: Chunkset<T>);
impl_index!(ChunkIndex: Chunk<T>);
impl_index!(ArchetypeIndex: ArchetypeData<T>);

impl Index<ComponentIndex> for [Entity] {
    type Output = Entity;
    #[inline(always)]
    fn index(&self, index: ComponentIndex) -> &Self::Output {
        &self[index.0]
    }
}
impl IndexMut<ComponentIndex> for [Entity] {
    #[inline(always)]
    fn index_mut(&mut self, index: ComponentIndex) -> &mut Self::Output {
        &mut self[index.0]
    }
}
impl Index<ComponentIndex> for Vec<Entity> {
    type Output = Entity;
    #[inline(always)]
    fn index(&self, index: ComponentIndex) -> &Self::Output {
        &self[index.0]
    }
}
impl IndexMut<ComponentIndex> for Vec<Entity> {
    #[inline(always)]
    fn index_mut(&mut self, index: ComponentIndex) -> &mut Self::Output {
        &mut self[index.0]
    }
}
impl Deref for ComponentIndex {
    type Target = usize;
    #[inline(always)]
    fn deref(&self) -> &usize {
        &self.0
    }
}
impl fmt::Display for ComponentIndex {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}
