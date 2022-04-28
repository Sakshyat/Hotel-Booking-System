from django.shortcuts import render
from .models import Review
from django.urls import reverse
from django.http import HttpResponse
from django.views.generic import CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin

# Create your views here.

class PostReview(LoginRequiredMixin, CreateView):
    model = Review
    template_name = 'reviews/review.html'
    fields = ['content']
    
    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

    def get_context_data(self, *args, **kwargs):
     context = super(PostReview, self).get_context_data(*args, **kwargs)            
     reviews= Review.objects.all().order_by('-date_posted')
     context["reviews"] = reviews
     return context

class EditReview(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Review
    fields = ['content']
    template_name = 'reviews/edit_review.html'
    
    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)
    
    def test_func(self):
        review = self.get_object()
        if self.request.user == review.author:
            return True
        return False

class DeleteReview(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Review
    success_url = '/reviews'

    def test_func(self):
        review = self.get_object()
        if self.request.user == review.author:
            return True
        return False